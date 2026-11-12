"""Held-out evaluation tasks, defined once and used by every method.

A *method* is anything that can score records against each other; the same
task definitions are applied to the trained encoder and to the baselines so
the numbers are directly comparable.

Cross-source retrieval
    The same perfume is described independently by Parfumo (note pyramid)
    and by the doevent table (flat ingredient list).  Given one description,
    find the other among every record of that source in the split.  This is
    the closest thing to ground truth for "these two note lists smell the
    same" that public data offers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import Corpus


@dataclass
class RetrievalTask:
    name: str
    query_indices: np.ndarray       # corpus indices of the queries
    gallery_indices: np.ndarray     # corpus indices of the gallery
    relevant: List[List[int]]       # per query: positions inside the gallery


def cross_source_tasks(
    corpus: Corpus, pairs: pd.DataFrame, split: str, full_gallery: bool = False
) -> Dict[str, RetrievalTask]:
    """Build both retrieval directions for one split.

    Queries are the unambiguous pairs of the split.  With ``full_gallery``
    the gallery is every record of the other source in *all* splits, which
    is harder and closer to how the product is used.
    """
    records = corpus.records
    in_split = (records["split"] == split).to_numpy()
    is_parfumo = (records["source"] == "parfumo").to_numpy()
    gallery_filter = np.ones(len(records), dtype=bool) if full_gallery else in_split
    doevent_gallery = np.flatnonzero(gallery_filter & ~is_parfumo)
    parfumo_gallery = np.flatnonzero(gallery_filter & is_parfumo)
    doevent_position = {index: position for position, index in enumerate(doevent_gallery)}
    parfumo_position = {index: position for position, index in enumerate(parfumo_gallery)}

    split_pairs = pairs[pairs["split"] == split]
    doevent_of_parfumo: Dict[int, List[int]] = {}
    parfumo_of_doevent: Dict[int, List[int]] = {}
    for doevent_id, parfumo_id in zip(split_pairs["doevent_id"], split_pairs["parfumo_id"]):
        doevent_index = corpus.index_of[doevent_id]
        parfumo_index = corpus.index_of[parfumo_id]
        doevent_of_parfumo.setdefault(parfumo_index, []).append(doevent_index)
        parfumo_of_doevent.setdefault(doevent_index, []).append(parfumo_index)

    unique_pairs = split_pairs[split_pairs["n_candidates"] == 1]
    parfumo_queries = sorted({corpus.index_of[value] for value in unique_pairs["parfumo_id"]})
    doevent_queries = sorted({corpus.index_of[value] for value in unique_pairs["doevent_id"]})
    suffix = "_full_gallery" if full_gallery else ""
    return {
        "parfumo_to_doevent" + suffix: RetrievalTask(
            "parfumo_to_doevent" + suffix,
            np.asarray(parfumo_queries),
            doevent_gallery,
            [
                [doevent_position[index] for index in doevent_of_parfumo[query]]
                for query in parfumo_queries
            ],
        ),
        "doevent_to_parfumo" + suffix: RetrievalTask(
            "doevent_to_parfumo" + suffix,
            np.asarray(doevent_queries),
            parfumo_gallery,
            [
                [parfumo_position[index] for index in parfumo_of_doevent[query]]
                for query in doevent_queries
            ],
        ),
    }


def accord_task_indices(corpus: Corpus, split: str) -> np.ndarray:
    in_split = (corpus.records["split"] == split).to_numpy()
    return np.flatnonzero(in_split & corpus.has_accords)


def family_task_indices(corpus: Corpus, split: str) -> np.ndarray:
    in_split = (corpus.records["split"] == split).to_numpy()
    return np.flatnonzero(in_split & (corpus.family >= 0) & (corpus.subfamily >= 0))
