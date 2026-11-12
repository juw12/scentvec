#!/usr/bin/env python3
"""Tabulate the refinement-pass ablations from their training logs.

    python3 scripts/ablation_table.py            # prints and writes reports/hone_ablations.md

Each run is summarised by its best validation epoch (the epoch the trainer
would keep).  Test data is never read here.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("h_full", "all inputs, 1 attention layer"),
    ("h_nochem", "without the odor-chemistry prior"),
    ("h_notok", "without note-word embeddings"),
    ("h_nopairs", "without cross-database pairs in training"),
    ("h_l0", "0 attention layers (pooled bag of notes)"),
    ("h_l2", "2 attention layers"),
]
COLUMNS = [
    ("val_score", "Selection score"),
    ("val_p2d_mrr", "MRR parfumo→doevent"),
    ("val_d2p_mrr", "MRR doevent→parfumo"),
    ("val_accord_map", "Accord mAP"),
    ("val_family_acc", "Family acc"),
    ("val_subfamily_acc", "Subfamily acc"),
]


def best_epoch(run_name: str):
    log_path = ROOT / "artifacts" / "runs" / run_name / "log.jsonl"
    if not log_path.exists():
        return None, 0
    epochs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not epochs:
        return None, 0
    return max(epochs, key=lambda entry: entry["val_score"]), len(epochs)


def main() -> None:
    reference, _ = best_epoch("h_full")
    lines = [
        "### Refinement-pass ablations (validation split, clean data, 8 epochs, seed 7)",
        "",
        "| Run | Change | Epochs | Best epoch | " + " | ".join(title for _, title in COLUMNS) + " |",
        "|---|---|---|---|" + "---|" * len(COLUMNS),
    ]
    for run_name, description in RUNS:
        best, count = best_epoch(run_name)
        if best is None:
            lines.append(f"| {run_name} | {description} | 0 | - |" + " - |" * len(COLUMNS))
            continue
        cells = []
        for key, _ in COLUMNS:
            cell = f"{best[key]:.4f}"
            if reference is not None and run_name != "h_full":
                cell += f" ({best[key] - reference[key]:+.4f})"
            cells.append(cell)
        lines.append(
            f"| {run_name} | {description} | {count} | {best['epoch']} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Differences in brackets are against `h_full`.  One seed per run: differences of a few",
        "thousandths are within run-to-run noise and should not be read as effects.",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    out = ROOT / "reports" / "hone_ablations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)


if __name__ == "__main__":
    main()
