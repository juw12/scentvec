"""Export the trained encoder for the in-browser demo on the landing page.

The landing page (docs/index.html) runs the real model in JavaScript.  It
needs far less than the PyTorch bundle:

* For the 1,053 notes that have their own id, the three per-note channels
  (note id, word tokens, chemistry profile) are folded into ONE vector per
  note.  That removes the token table, the chemistry table and the chemistry
  projection from the download.
* The class prototypes are stored unit-length and pre-multiplied by their
  learned scale, so a head is a plain matrix product plus a bias.
* Everything is stored as little-endian float16.

Output:

    docs/model.bin    all tensors, concatenated
    docs/model.json   tensor directory, vocabulary, labels and reference
                      outputs that the page checks itself against on load

Notes without their own id are NOT covered by the browser demo; the Python
package handles those through their words and chemistry.

    ./.venv/bin/python scripts/export_web_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scentvec.encoder import NoteEncoder  # noqa: E402
from scentvec.features import PAD  # noqa: E402

FORMAT_VERSION = 1

# Reference pyramids: the page recomputes these on load and reports how close
# the JavaScript port is to the PyTorch model.
REFERENCE_PYRAMIDS = [
    {
        "top": ["bergamot", "lemon", "neroli"],
        "middle": ["lavender", "rosemary", "petitgrain"],
        "base": ["musk", "cedarwood"],
    },
    {
        "top": ["cinnamon", "rum"],
        "middle": ["tobacco", "vanilla"],
        "base": ["tonka bean", "benzoin", "incense"],
    },
    {
        "top": ["saffron"],
        "middle": ["rose", "geranium"],
        "base": ["oud", "patchouli", "amber"],
    },
    {
        "top": [],
        "middle": ["jasmine"],
        "base": [],
    },
    {
        "top": ["grapefruit", "sea salt", "mint"],
        "middle": ["sage", "algae"],
        "base": ["ambergris", "driftwood", "white musk"],
    },
]


def fold_note_vectors(encoder: NoteEncoder) -> np.ndarray:
    """One input vector per vocabulary note: id + mean of words + chemistry."""
    model = encoder.model
    note_table = encoder._note_table
    notes = encoder.vocabulary.notes
    rows = np.asarray([note_table.row(note) for note in notes], dtype=np.int64)
    note_ids, token_ids, chem = note_table.arrays()
    with torch.no_grad():
        ids = torch.from_numpy(note_ids[rows])
        tokens = torch.from_numpy(token_ids[rows])
        profile = torch.from_numpy(chem[rows])
        token_count = (tokens != PAD).sum(dim=-1, keepdim=True).clamp(min=1)
        folded = (
            model.note_embedding(ids)
            + model.token_embedding(tokens).sum(dim=-2) / token_count
            + model.chem_projection(profile)
        )
    return folded.numpy()


def collect_tensors(encoder: NoteEncoder) -> Dict[str, np.ndarray]:
    model = encoder.model
    config = model.config
    if config.n_layers != 1:
        raise SystemExit("The browser port implements exactly one attention layer.")
    if not (config.use_note_ids and config.use_tokens and config.use_chem and config.use_positions):
        raise SystemExit("The browser port expects the full model (all input channels on).")

    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    layer = "context.layers.0."
    tensors = {
        "note_vectors": fold_note_vectors(encoder),
        "position": state["position_embedding.weight"],
        "input_norm_weight": state["input_norm.weight"],
        "input_norm_bias": state["input_norm.bias"],
        "attn_in_weight": state[layer + "self_attn.in_proj_weight"],
        "attn_in_bias": state[layer + "self_attn.in_proj_bias"],
        "attn_out_weight": state[layer + "self_attn.out_proj.weight"],
        "attn_out_bias": state[layer + "self_attn.out_proj.bias"],
        "ff1_weight": state[layer + "linear1.weight"],
        "ff1_bias": state[layer + "linear1.bias"],
        "ff2_weight": state[layer + "linear2.weight"],
        "ff2_bias": state[layer + "linear2.bias"],
        "norm1_weight": state[layer + "norm1.weight"],
        "norm1_bias": state[layer + "norm1.bias"],
        "norm2_weight": state[layer + "norm2.weight"],
        "norm2_bias": state[layer + "norm2.bias"],
        "output_norm_weight": state["output_norm.weight"],
        "output_norm_bias": state["output_norm.bias"],
        "pool_score_weight": state["attention_score.weight"],
        "pool_score_bias": state["attention_score.bias"],
        "proj1_weight": state["projection.0.weight"],
        "proj1_bias": state["projection.0.bias"],
        "proj2_weight": state["projection.2.weight"],
        "proj2_bias": state["projection.2.bias"],
    }
    for name, head in [
        ("accord", model.accord_head),
        ("family", model.family_head),
        ("subfamily", model.subfamily_head),
    ]:
        with torch.no_grad():
            scaled = head.log_scale.exp() * F.normalize(head.prototypes, dim=-1)
        tensors[name + "_weight"] = scaled.numpy()
        tensors[name + "_bias"] = head.bias.detach().numpy()
    return tensors


def reference_outputs(encoder: NoteEncoder) -> List[dict]:
    references = []
    for pyramid in REFERENCE_PYRAMIDS:
        for notes in pyramid.values():
            for note in notes:
                if note not in encoder.vocabulary.note_to_id:
                    raise SystemExit(f"Reference note {note!r} is not in the vocabulary.")
        vector = encoder.embed(**pyramid)
        scores = encoder.head_scores(vector)
        references.append(
            {
                "pyramid": pyramid,
                "embedding": [round(float(value), 6) for value in vector],
                "accords": [round(float(value), 5) for value in scores["accords"][0]],
                "family": [round(float(value), 5) for value in scores["family"][0]],
            }
        )
    return references


def export(model_dir: Path, output_dir: Path) -> dict:
    encoder = NoteEncoder.load(model_dir)
    tensors = collect_tensors(encoder)

    directory = {}
    chunks = []
    offset = 0
    for name, array in tensors.items():
        half = np.ascontiguousarray(array, dtype="<f2")
        if not np.isfinite(half.astype(np.float32)).all():
            raise SystemExit(f"Tensor {name} does not survive float16.")
        directory[name] = {"shape": list(half.shape), "offset": offset, "length": int(half.size)}
        chunks.append(half.tobytes())
        offset += int(half.size)
    blob = b"".join(chunks)

    config = encoder.model.config
    bundle = json.loads((model_dir / "bundle.json").read_text())
    manifest = {
        "format": FORMAT_VERSION,
        "dtype": "float16-le",
        "binary": "model.bin",
        "binary_bytes": len(blob),
        "binary_sha256": hashlib.sha256(blob).hexdigest(),
        "model_version": bundle["scentvec_version"],
        "hidden_dim": config.hidden_dim,
        "embed_dim": config.embed_dim,
        "n_heads": config.n_heads,
        "max_notes": 40,
        "layer_norm_eps": 1e-5,
        "positions": ["top", "middle", "base", "unknown"],
        "notes": encoder.vocabulary.notes,
        "accords": encoder.vocabulary.accords,
        "families": encoder.vocabulary.families,
        "subfamilies": encoder.vocabulary.subfamilies,
        "tensors": directory,
        "reference": reference_outputs(encoder),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.bin").write_bytes(blob)
    (output_dir / "model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "model")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs")
    arguments = parser.parse_args()
    manifest = export(arguments.model_dir, arguments.output_dir)
    print(
        f"wrote {arguments.output_dir / 'model.bin'} "
        f"({manifest['binary_bytes'] / 1e6:.2f} MB, {len(manifest['tensors'])} tensors, "
        f"{len(manifest['notes'])} notes)"
    )


if __name__ == "__main__":
    main()
