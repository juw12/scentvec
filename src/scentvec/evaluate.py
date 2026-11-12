"""Evaluate a trained run against the baselines on a held-out split.

Run:  python -m scentvec.evaluate --run final --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .baselines import BaselineSuite
from .data import PROCESSED_DIR, PROJECT_ROOT
from .features import Corpus, Vocabulary, load_corpus
from .metrics import (
    accord_ranking_metrics,
    best_relevant_ranks,
    classification_metrics,
    retrieval_summary,
)
from .tasks import accord_task_indices, cross_source_tasks, family_task_indices
from .train import evaluate_model, load_run, resolve_run_dir

RETRIEVAL_BASELINES = ["jaccard", "tfidf_notes", "tfidf_words", "lsa"]
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_baselines(
    corpus: Corpus, pairs: pd.DataFrame, split: str, full_gallery: bool
) -> Dict[str, dict]:
    suite = BaselineSuite(corpus)
    tasks = cross_source_tasks(corpus, pairs, split, full_gallery=False)
    if full_gallery:
        tasks.update(cross_source_tasks(corpus, pairs, split, full_gallery=True))

    results: Dict[str, dict] = {method: {} for method in RETRIEVAL_BASELINES}
    for method in RETRIEVAL_BASELINES:
        for name, task in tasks.items():
            similarity = suite.similarity(method, task.query_indices, task.gallery_indices)
            summary = retrieval_summary(best_relevant_ranks(similarity, task.relevant))
            summary["gallery_size"] = int(len(task.gallery_indices))
            results[method][name] = summary

    suite.fit_label_models()
    accord_indices = accord_task_indices(corpus, split)
    family_indices = family_task_indices(corpus, split)
    vocabulary = corpus.note_table.vocabulary
    results["logreg_bag_of_notes"] = {
        "accords": accord_ranking_metrics(
            suite.accord_scores(accord_indices), corpus.accord_targets[accord_indices]
        ),
        "family": classification_metrics(
            suite.class_scores("family", family_indices, len(vocabulary.families)),
            corpus.family[family_indices],
        ),
        "subfamily": classification_metrics(
            suite.class_scores("subfamily", family_indices, len(vocabulary.subfamilies)),
            corpus.subfamily[family_indices],
        ),
    }
    return results


def format_report(report: dict) -> str:
    """Markdown tables for the README and model card."""
    methods = report["methods"]
    lines: List[str] = [f"### Held-out results ({report['split']} split)", ""]
    retrieval_tasks = [
        name for name in next(iter(methods.values())) if name.startswith(("parfumo_", "doevent_"))
    ]
    for task in retrieval_tasks:
        example = next(result[task] for result in methods.values() if task in result)
        lines.append(
            f"**{task}** - {example['n_queries']} queries, gallery of "
            f"{example['gallery_size']:,} records"
        )
        lines.append("")
        lines.append("| Method | R@1 | R@5 | R@10 | MRR | Median rank |")
        lines.append("|---|---|---|---|---|---|")
        for method, result in methods.items():
            if task not in result:
                continue
            row = result[task]
            lines.append(
                f"| {method} | {row['recall@1']:.3f} | {row['recall@5']:.3f} | "
                f"{row['recall@10']:.3f} | {row['mrr']:.3f} | {row['median_rank']:.0f} |"
            )
        lines.append("")
    lines.append("**Label prediction from notes**")
    lines.append("")
    lines.append(
        "| Method | Accord mAP | Accord P@1 | Accord nDCG@5 | Family acc | Subfamily acc | "
        "Subfamily macro-F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for method, result in methods.items():
        if "accords" not in result:
            continue
        lines.append(
            f"| {method} | {result['accords']['map']:.3f} | {result['accords']['precision@1']:.3f} "
            f"| {result['accords']['ndcg@5']:.3f} | {result['family']['accuracy']:.3f} | "
            f"{result['subfamily']['accuracy']:.3f} | {result['subfamily']['macro_f1']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_run(
    run_dir: Path,
    split: str,
    full_gallery: bool = True,
    with_baselines: bool = True,
    processed_dir: Path = PROCESSED_DIR,
) -> dict:
    vocabulary = Vocabulary.load(run_dir / "vocab.json")
    corpus, pairs, _, _ = load_corpus(processed_dir, vocabulary)
    model, _ = load_run(run_dir)
    methods: Dict[str, dict] = {
        "scentvec": evaluate_model(model, corpus, pairs, split, full_gallery=full_gallery)
    }
    if with_baselines:
        methods.update(evaluate_baselines(corpus, pairs, split, full_gallery))
    return {"run": run_dir.name, "split": split, "methods": methods}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run against the baselines.")
    parser.add_argument(
        "--run", "--run-dir", dest="run", required=True, help="run name or run directory"
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)

    try:
        run_dir = resolve_run_dir(arguments.run)
    except FileNotFoundError as error:
        raise SystemExit(f"scentvec evaluate: {error}")
    report = evaluate_run(run_dir, arguments.split, with_baselines=not arguments.no_baselines)
    markdown = format_report(report)
    print(markdown)
    out = arguments.out or REPORTS_DIR / f"eval_{run_dir.name}_{arguments.split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".md").write_text(markdown)


if __name__ == "__main__":
    main()
