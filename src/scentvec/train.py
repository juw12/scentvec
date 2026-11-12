"""Train the perfume encoder.

Three signals shape the space:

1. Contrastive (InfoNCE).  Two descriptions of the same perfume must land
   together and away from every other perfume in the batch.  Where the two
   databases overlap the positives are the two independent descriptions;
   elsewhere they are two randomly thinned copies of one note list.
2. Crowd-voted accords from Parfumo (ranked, graded targets).
3. Expert olfactive family and subfamily from the doevent table.

Run:  python -m scentvec.train --run-name base
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import PROCESSED_DIR, PROJECT_ROOT
from .features import POSITION_UNKNOWN, Corpus, load_corpus, pad_batch
from .metrics import (
    accord_ranking_metrics,
    best_relevant_ranks,
    classification_metrics,
    retrieval_summary,
)
from .model import ModelConfig, ScentEncoder, to_tensors
from .tasks import accord_task_indices, cross_source_tasks, family_task_indices

RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"


@dataclass
class TrainConfig:
    run_name: str = "base"
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    temperature: float = 0.07
    accord_weight: float = 1.0
    family_weight: float = 0.5
    contrastive_weight: float = 1.0
    max_note_dropout: float = 0.4
    position_dropout: float = 0.3
    use_cross_source_pairs: bool = True
    seed: int = 7
    threads: int = 4


@dataclass
class Entity:
    """One perfume as the trainer sees it: two sides that must embed together."""

    side_a: List[int]
    side_b: List[int]


def build_entities(corpus: Corpus, pairs: pd.DataFrame, use_pairs: bool) -> List[Entity]:
    records = corpus.records
    train_indices = np.flatnonzero((records["split"] == "train").to_numpy())
    entities: List[Entity] = []
    paired = set()
    if use_pairs:
        train_pairs = pairs[pairs["split"] == "train"]
        for doevent_id, group in train_pairs.groupby("doevent_id"):
            doevent_index = corpus.index_of[doevent_id]
            parfumo_indices = [corpus.index_of[value] for value in group["parfumo_id"]]
            entities.append(Entity(parfumo_indices, [doevent_index]))
            paired.add(doevent_index)
            paired.update(parfumo_indices)
    for index in train_indices:
        if index not in paired:
            entities.append(Entity([int(index)], [int(index)]))
    return entities


class Augmenter:
    def __init__(self, corpus: Corpus, config: TrainConfig, generator: np.random.Generator) -> None:
        self.corpus = corpus
        self.config = config
        self.generator = generator

    def view(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """A randomly thinned copy of one record's notes."""
        rows = self.corpus.rows[index]
        positions = self.corpus.positions[index]
        length = len(rows)
        if length > 2:
            drop_rate = self.generator.uniform(0.0, self.config.max_note_dropout)
            keep = self.generator.random(length) >= drop_rate
            if keep.sum() < 2:
                keep[self.generator.choice(length, size=2, replace=False)] = True
            rows = rows[keep]
            positions = positions[keep]
        if self.generator.random() < self.config.position_dropout:
            # Pretend the pyramid is unknown, as it is for flat ingredient lists.
            positions = np.full_like(positions, POSITION_UNKNOWN)
        return rows, positions


def contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, groups: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = first @ second.T / temperature
    # Other entries of the same perfume line (EDT vs EDP, duplicates) are not
    # real negatives; take them out of the denominator.
    same_group = groups.unsqueeze(0) == groups.unsqueeze(1)
    same_group.fill_diagonal_(False)
    logits = logits.masked_fill(same_group, -1e9)
    targets = torch.arange(first.shape[0])
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


@torch.no_grad()
def embed_corpus(
    model: ScentEncoder, corpus: Corpus, indices: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(indices), batch_size):
        batch = to_tensors(corpus.batch(indices[start : start + batch_size]))
        chunks.append(model(**batch).numpy())
    if not chunks:
        return np.zeros((0, model.config.embed_dim), dtype=np.float32)
    return np.concatenate(chunks)


@torch.no_grad()
def evaluate_model(
    model: ScentEncoder,
    corpus: Corpus,
    pairs: pd.DataFrame,
    split: str,
    full_gallery: bool = False,
) -> Dict[str, dict]:
    """Every held-out task for the encoder on one split."""
    all_indices = np.arange(len(corpus.records))
    needed = all_indices if full_gallery else np.flatnonzero(
        (corpus.records["split"] == split).to_numpy()
    )
    vectors = np.zeros((len(corpus.records), model.config.embed_dim), dtype=np.float32)
    vectors[needed] = embed_corpus(model, corpus, needed)

    results: Dict[str, dict] = {}
    tasks = cross_source_tasks(corpus, pairs, split, full_gallery=False)
    if full_gallery:
        tasks.update(cross_source_tasks(corpus, pairs, split, full_gallery=True))
    for name, task in tasks.items():
        similarity = vectors[task.query_indices] @ vectors[task.gallery_indices].T
        summary = retrieval_summary(best_relevant_ranks(similarity, task.relevant))
        summary["gallery_size"] = int(len(task.gallery_indices))
        results[name] = summary

    heads = model.heads(torch.from_numpy(vectors))
    accord_indices = accord_task_indices(corpus, split)
    results["accords"] = accord_ranking_metrics(
        heads["accords"].numpy()[accord_indices], corpus.accord_targets[accord_indices]
    )
    family_indices = family_task_indices(corpus, split)
    results["family"] = classification_metrics(
        heads["family"].numpy()[family_indices], corpus.family[family_indices]
    )
    results["subfamily"] = classification_metrics(
        heads["subfamily"].numpy()[family_indices], corpus.subfamily[family_indices]
    )
    return results


def selection_score(results: Dict[str, dict]) -> float:
    """Single number used to pick the checkpoint: retrieval first, labels second."""
    retrieval = 0.5 * (
        results["parfumo_to_doevent"]["mrr"] + results["doevent_to_parfumo"]["mrr"]
    )
    labels = (results["accords"]["map"] + results["subfamily"]["accuracy"]) / 2.0
    return 0.6 * retrieval + 0.4 * labels


def train(config: TrainConfig, model_overrides: Optional[dict] = None) -> Path:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)

    run_dir = RUNS_DIR / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus, pairs, vocabulary, chem_table = load_corpus(PROCESSED_DIR)
    vocabulary.save(run_dir / "vocab.json")

    model_config = ModelConfig(
        n_note_ids=vocabulary.n_note_ids,
        n_token_ids=vocabulary.n_token_ids,
        chem_dim=chem_table.dim,
        n_accords=len(vocabulary.accords),
        n_families=len(vocabulary.families),
        n_subfamilies=len(vocabulary.subfamilies),
        **(model_overrides or {}),
    )
    model = ScentEncoder(model_config)
    (run_dir / "config.json").write_text(
        json.dumps({"model": model_config.to_json(), "train": asdict(config)}, indent=2)
    )

    entities = build_entities(corpus, pairs, config.use_cross_source_pairs)
    steps_per_epoch = len(entities) // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def learning_rate_factor(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    augmenter = Augmenter(corpus, config, generator)
    accord_targets = torch.from_numpy(corpus.accord_targets)
    has_accords = torch.from_numpy(corpus.has_accords)
    family = torch.from_numpy(corpus.family)
    subfamily = torch.from_numpy(corpus.subfamily)
    groups = torch.from_numpy(corpus.group)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"run={config.run_name} entities={len(entities)} steps/epoch={steps_per_epoch} "
        f"parameters={parameter_count:,}",
        flush=True,
    )

    best_score = -1.0
    log_path = run_dir / "log.jsonl"
    log_path.write_text("")
    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.time()
        order = generator.permutation(len(entities))
        running = {"total": 0.0, "contrastive": 0.0, "accords": 0.0, "family": 0.0}
        for step in range(steps_per_epoch):
            batch_entities = [
                entities[position]
                for position in order[step * config.batch_size : (step + 1) * config.batch_size]
            ]
            index_a = [int(generator.choice(entity.side_a)) for entity in batch_entities]
            index_b = [int(generator.choice(entity.side_b)) for entity in batch_entities]
            views = [augmenter.view(index) for index in index_a + index_b]
            batch = to_tensors(
                pad_batch(
                    corpus.note_table,
                    [rows for rows, _ in views],
                    [positions for _, positions in views],
                )
            )
            embeddings = model(**batch)
            batch_count = len(batch_entities)
            first, second = embeddings[:batch_count], embeddings[batch_count:]
            index_a_tensor = torch.tensor(index_a)
            index_b_tensor = torch.tensor(index_b)

            loss_contrastive = contrastive_loss(
                first, second, groups[index_a_tensor], config.temperature
            )

            # Labels are properties of the perfume, so both descriptions of a
            # matched pair are trained to predict them.
            entity_has_accords = has_accords[index_a_tensor] | has_accords[index_b_tensor]
            entity_accords = torch.maximum(
                accord_targets[index_a_tensor], accord_targets[index_b_tensor]
            )
            entity_family = torch.maximum(family[index_a_tensor], family[index_b_tensor])
            entity_subfamily = torch.maximum(
                subfamily[index_a_tensor], subfamily[index_b_tensor]
            )
            loss_accords = torch.zeros(())
            loss_family = torch.zeros(())
            for side in (first, second):
                heads = model.heads(side)
                if entity_has_accords.any():
                    loss_accords = loss_accords + 0.5 * F.binary_cross_entropy_with_logits(
                        heads["accords"][entity_has_accords], entity_accords[entity_has_accords]
                    )
                labelled = entity_family >= 0
                if labelled.any():
                    loss_family = loss_family + 0.25 * F.cross_entropy(
                        heads["family"][labelled], entity_family[labelled]
                    )
                labelled = entity_subfamily >= 0
                if labelled.any():
                    loss_family = loss_family + 0.25 * F.cross_entropy(
                        heads["subfamily"][labelled], entity_subfamily[labelled]
                    )

            loss = (
                config.contrastive_weight * loss_contrastive
                + config.accord_weight * loss_accords
                + config.family_weight * loss_family
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running["total"] += float(loss)
            running["contrastive"] += float(loss_contrastive)
            running["accords"] += float(loss_accords)
            running["family"] += float(loss_family)

        validation = evaluate_model(model, corpus, pairs, "val")
        score = selection_score(validation)
        entry = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "loss": {name: round(value / steps_per_epoch, 4) for name, value in running.items()},
            "val_score": round(score, 4),
            "val_p2d_mrr": round(validation["parfumo_to_doevent"]["mrr"], 4),
            "val_d2p_mrr": round(validation["doevent_to_parfumo"]["mrr"], 4),
            "val_accord_map": round(validation["accords"]["map"], 4),
            "val_family_acc": round(validation["family"]["accuracy"], 4),
            "val_subfamily_acc": round(validation["subfamily"]["accuracy"], 4),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), run_dir / "model.pt")
            (run_dir / "val_metrics.json").write_text(
                json.dumps({"epoch": epoch, "score": score, "metrics": validation}, indent=2)
            )
    print(f"best val score {best_score:.4f} -> {run_dir}", flush=True)
    return run_dir


def resolve_run_dir(reference: str) -> Path:
    """Turn a run name ('final') or a run directory into the directory of a finished run."""
    candidates = [Path(reference), RUNS_DIR / str(reference)]
    for candidate in candidates:
        if (candidate / "model.pt").exists():
            return candidate
    names = sorted(path.parent.name for path in RUNS_DIR.glob("*/model.pt"))
    known = ", ".join(names) if names else "none yet; run `scentvec train` first"
    raise FileNotFoundError(f"no finished training run '{reference}' (available: {known})")


def load_run(run_dir: Path) -> Tuple[ScentEncoder, dict]:
    """Load a trained encoder and its saved configuration."""
    saved = json.loads((run_dir / "config.json").read_text())
    model = ScentEncoder(ModelConfig(**saved["model"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, saved


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the scentvec encoder.")
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, type=lambda text: text.lower() in ("1", "true", "yes"), default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--note-unk-dropout", type=float, default=0.1)
    for switch in ["use-note-ids", "use-tokens", "use-chem", "use-positions"]:
        parser.add_argument(
            "--" + switch, type=lambda text: text.lower() in ("1", "true", "yes"), default=True
        )
    arguments = vars(parser.parse_args(argv))
    model_keys = [
        "hidden_dim", "embed_dim", "n_layers", "dropout", "note_unk_dropout",
        "use_note_ids", "use_tokens", "use_chem", "use_positions",
    ]
    model_overrides = {key: arguments.pop(key) for key in model_keys}
    train(TrainConfig(**arguments), model_overrides)


if __name__ == "__main__":
    main()
