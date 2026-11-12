#!/usr/bin/env python3
"""Is the retrieval gain over the baselines larger than sampling noise?

    PYTHONPATH=src ./.venv/bin/python scripts/paired_bootstrap.py --run final --split test

For every cross-database retrieval task this computes the reciprocal rank of
each query under scentvec and under every baseline, then resamples the queries
(paired, with replacement) to get a 95% interval for the difference in MRR.
The interval covers query sampling only: it says nothing about how much a
different training seed would move the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scentvec.baselines import BaselineSuite
from scentvec.data import PROCESSED_DIR
from scentvec.evaluate import REPORTS_DIR, RETRIEVAL_BASELINES
from scentvec.features import Vocabulary, load_corpus
from scentvec.metrics import best_relevant_ranks
from scentvec.tasks import cross_source_tasks
from scentvec.train import embed_corpus, load_run, resolve_run_dir


def reciprocal_ranks(run_dir: Path, split: str) -> Dict[str, Dict[str, np.ndarray]]:
    """task -> method -> reciprocal rank per query (same query order for every method)."""
    vocabulary = Vocabulary.load(run_dir / "vocab.json")
    corpus, pairs, _, _ = load_corpus(PROCESSED_DIR, vocabulary)
    model, _ = load_run(run_dir)
    all_indices = np.arange(len(corpus.records))
    vectors = embed_corpus(model, corpus, all_indices)
    suite = BaselineSuite(corpus)

    tasks = cross_source_tasks(corpus, pairs, split, full_gallery=False)
    tasks.update(cross_source_tasks(corpus, pairs, split, full_gallery=True))
    table: Dict[str, Dict[str, np.ndarray]] = {}
    for name, task in tasks.items():
        similarity = vectors[task.query_indices] @ vectors[task.gallery_indices].T
        per_method = {"scentvec": 1.0 / best_relevant_ranks(similarity, task.relevant)}
        for method in RETRIEVAL_BASELINES:
            similarity = suite.similarity(method, task.query_indices, task.gallery_indices)
            per_method[method] = 1.0 / best_relevant_ranks(similarity, task.relevant)
        table[name] = per_method
    return table


def paired_interval(
    ours: np.ndarray, theirs: np.ndarray, resamples: int, generator: np.random.Generator
) -> Dict[str, float]:
    difference = ours - theirs
    draws = generator.integers(0, len(difference), size=(resamples, len(difference)))
    means = difference[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mrr_ours": float(ours.mean()),
        "mrr_baseline": float(theirs.mean()),
        "difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "share_of_resamples_above_zero": float((means > 0).mean()),
        "queries_better": int((difference > 0).sum()),
        "queries_worse": int((difference < 0).sum()),
        "queries_equal": int((difference == 0).sum()),
    }


def format_markdown(report: dict) -> str:
    lines: List[str] = [
        f"### Paired bootstrap of the MRR difference ({report['split']} split, "
        f"{report['resamples']:,} resamples)",
        "",
        "Difference = scentvec minus baseline, same queries.  The interval covers query "
        "sampling only, not training-seed variation.",
        "",
    ]
    for task, rows in report["tasks"].items():
        n_queries = report["queries"][task]
        lines.append(f"**{task}** - {n_queries} queries")
        lines.append("")
        lines.append("| Baseline | scentvec MRR | Baseline MRR | Difference | 95% interval | Better / worse / equal |")
        lines.append("|---|---|---|---|---|---|")
        for method, row in rows.items():
            lines.append(
                f"| {method} | {row['mrr_ours']:.3f} | {row['mrr_baseline']:.3f} | "
                f"{row['difference']:+.3f} | {row['ci95_low']:+.3f} to {row['ci95_high']:+.3f} | "
                f"{row['queries_better']} / {row['queries_worse']} / {row['queries_equal']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="run name or run directory")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args(argv)

    run_dir = resolve_run_dir(arguments.run)
    table = reciprocal_ranks(run_dir, arguments.split)
    generator = np.random.default_rng(arguments.seed)
    report = {
        "run": run_dir.name,
        "split": arguments.split,
        "resamples": arguments.resamples,
        "queries": {task: int(len(methods["scentvec"])) for task, methods in table.items()},
        "tasks": {
            task: {
                method: paired_interval(
                    methods["scentvec"], methods[method], arguments.resamples, generator
                )
                for method in RETRIEVAL_BASELINES
            }
            for task, methods in table.items()
        },
    }
    markdown = format_markdown(report)
    print(markdown)
    out = REPORTS_DIR / f"significance_{run_dir.name}_{arguments.split}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".md").write_text(markdown)


if __name__ == "__main__":
    main()
