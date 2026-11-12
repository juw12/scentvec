"""Metric functions shared by training-time validation and the final report."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def best_relevant_ranks(similarity: np.ndarray, relevant: Sequence[Sequence[int]]) -> np.ndarray:
    """1-based rank of the best-placed relevant gallery item for every query.

    Ties are counted against the method (a tied wrong item ranks first), so a
    scorer cannot look good by returning many identical scores.
    """
    ranks = np.empty(len(relevant), dtype=np.int64)
    for query, relevant_items in enumerate(relevant):
        scores = similarity[query]
        best = scores[list(relevant_items)].max()
        ranks[query] = int((scores >= best).sum()) - (
            int((scores[list(relevant_items)] == best).sum()) - 1
        )
    return ranks


def retrieval_summary(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "n_queries": int(len(ranks)),
        "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
    }


def accord_ranking_metrics(scores: np.ndarray, gains: np.ndarray) -> Dict[str, float]:
    """Ranking quality of predicted accords against the crowd-voted list.

    ``gains`` holds graded relevance (0 = accord not listed).  Reports mean
    average precision over perfumes, precision of the top prediction, and
    nDCG@5 using the vote rank as the gain.
    """
    order = np.argsort(-scores, axis=1)
    relevant = gains > 0
    average_precisions: List[float] = []
    ndcgs: List[float] = []
    top_hits: List[float] = []
    discounts = 1.0 / np.log2(np.arange(2, 7))
    for row in range(scores.shape[0]):
        ranked_relevant = relevant[row, order[row]]
        hits = np.cumsum(ranked_relevant)
        precision_at_hits = hits[ranked_relevant] / (np.flatnonzero(ranked_relevant) + 1)
        average_precisions.append(float(precision_at_hits.mean()))
        top_hits.append(float(ranked_relevant[0]))
        dcg = float((gains[row, order[row, :5]] * discounts).sum())
        ideal = np.sort(gains[row])[::-1][:5]
        ndcgs.append(dcg / float((ideal * discounts).sum()))
    return {
        "n_perfumes": int(scores.shape[0]),
        "map": float(np.mean(average_precisions)),
        "precision@1": float(np.mean(top_hits)),
        "ndcg@5": float(np.mean(ndcgs)),
    }


def classification_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    predictions = scores.argmax(axis=1)
    classes = np.unique(labels)
    per_class_f1 = []
    for label in classes:
        true_positive = float(((predictions == label) & (labels == label)).sum())
        predicted = float((predictions == label).sum())
        actual = float((labels == label).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        per_class_f1.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "n_perfumes": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(np.mean(per_class_f1)),
    }
