"""Build the searchable perfume catalog and its embedding index.

One catalog item is one perfume as a shopper would think of it.  Parfumo
entries are the backbone; the matching doevent entry (same brand, name and
year) is folded in, contributing family, gender, description and a second
opinion on the notes.  doevent entries without a Parfumo match become items
of their own.

Run:  python -m scentvec.catalog --model-dir artifacts/model
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import PROCESSED_DIR, PROJECT_ROOT
from .encoder import NoteEncoder
from .features import record_notes_with_positions

MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
INDEX_DIR = PROJECT_ROOT / "artifacts" / "index"
TOP_PREDICTED_ACCORDS = 5


def _first_text(values: List[Optional[str]]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _merge_notes(primary: List[str], secondary: List[str]) -> List[str]:
    return list(primary) + [note for note in secondary if note not in primary]


def build_catalog(encoder: NoteEncoder, processed_dir: Path = PROCESSED_DIR) -> Dict[str, object]:
    records = pd.read_parquet(processed_dir / "records.parquet")
    records = records[records["n_notes"] > 0].reset_index(drop=True)
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    index_of = {record_id: index for index, record_id in enumerate(records["record_id"])}
    rows = records.to_dict("records")

    encoded = [record_notes_with_positions(row) for row in rows]
    record_vectors = encoder.encode(encoded)

    linked_doevent: Dict[int, List[int]] = defaultdict(list)
    unique_link: Dict[int, List[int]] = defaultdict(list)
    doevent_with_parfumo = set()
    for doevent_id, parfumo_id, n_candidates in zip(
        pairs["doevent_id"], pairs["parfumo_id"], pairs["n_candidates"]
    ):
        parfumo_index = index_of[parfumo_id]
        doevent_index = index_of[doevent_id]
        linked_doevent[parfumo_index].append(doevent_index)
        doevent_with_parfumo.add(doevent_index)
        if n_candidates == 1:
            unique_link[parfumo_index].append(doevent_index)

    items: List[dict] = []
    vectors: List[np.ndarray] = []
    seen_identity = set()

    def add_item(primary: int, others: List[int], blend_with: List[int]) -> None:
        row = rows[primary]
        other_rows = [rows[index] for index in others]
        pyramid_notes, _ = encoded[primary]
        all_notes = list(pyramid_notes)
        for other in other_rows:
            all_notes = _merge_notes(all_notes, list(other["notes_flat"]))
        # Identical listings (same line, same notes) are scraper duplicates.
        identity = (row["brand_key"], row["name_key"], row["concentration"], frozenset(all_notes))
        if identity in seen_identity:
            return
        seen_identity.add(identity)

        vector = record_vectors[[primary] + blend_with].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return
        vectors.append(vector / norm)
        year = row["year"]
        if pd.isna(year):
            year = next(
                (other["year"] for other in other_rows if not pd.isna(other["year"])), None
            )
        rating = row["rating"]
        items.append(
            {
                "brand": row["brand"],
                "name": row["name"],
                "brand_key": row["brand_key"],
                "name_key": row["name_key"],
                "year": None if year is None or pd.isna(year) else int(year),
                "concentration": row["concentration"],
                "gender": _first_text([row["gender"]] + [other["gender"] for other in other_rows]),
                "notes_top": list(row["notes_top"]),
                "notes_middle": list(row["notes_middle"]),
                "notes_base": list(row["notes_base"]),
                "notes_other": [note for note in all_notes if note not in pyramid_notes]
                if row["source"] == "parfumo"
                else list(row["notes_flat"]),
                "notes_all": all_notes,
                "accords_voted": list(row["accords"]),
                "family": _first_text([row["family"]] + [o["family"] for o in other_rows]),
                "subfamily": _first_text([row["subfamily"]] + [o["subfamily"] for o in other_rows]),
                "rating": None if rating is None or pd.isna(rating) else float(rating),
                "rating_count": int(row["rating_count"]),
                "perfumers": list(row["perfumers"]),
                "url": row["url"],
                "description": _first_text(
                    [row["description"]] + [other["description"] for other in other_rows]
                ),
                "sources": sorted({row["source"]} | {other["source"] for other in other_rows}),
                "record_ids": [row["record_id"]] + [other["record_id"] for other in other_rows],
                "split": row["split"],
            }
        )

    for index, row in enumerate(rows):
        if row["source"] == "parfumo":
            add_item(index, linked_doevent.get(index, []), unique_link.get(index, []))
    for index, row in enumerate(rows):
        if row["source"] == "doevent" and index not in doevent_with_parfumo:
            add_item(index, [], [])

    embeddings = np.vstack(vectors).astype(np.float32)
    scores = encoder.head_scores(embeddings)
    vocabulary = encoder.vocabulary
    for position, item in enumerate(items):
        item["item_id"] = position
        order = np.argsort(-scores["accords"][position])[:TOP_PREDICTED_ACCORDS]
        item["accords_predicted"] = [vocabulary.accords[index] for index in order]
        item["accords_predicted_scores"] = [
            round(float(scores["accords"][position, index]), 3) for index in order
        ]
        family_index = int(scores["family"][position].argmax())
        subfamily_index = int(scores["subfamily"][position].argmax())
        item["family_predicted"] = vocabulary.families[family_index]
        item["subfamily_predicted"] = vocabulary.subfamilies[subfamily_index]
    return {
        "catalog": pd.DataFrame(items),
        "embeddings": embeddings,
        "accord_scores": scores["accords"].astype(np.float16),
    }


def build_index(
    model_dir: Path = MODEL_DIR, out_dir: Path = INDEX_DIR, processed_dir: Path = PROCESSED_DIR
) -> dict:
    encoder = NoteEncoder.load(model_dir)
    built = build_catalog(encoder, processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog: pd.DataFrame = built["catalog"]
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    np.save(out_dir / "embeddings.npy", built["embeddings"])
    np.save(out_dir / "accord_scores.npy", built["accord_scores"])
    summary = {
        "items": int(len(catalog)),
        "embedding_dim": int(built["embeddings"].shape[1]),
        "items_with_both_sources": int((catalog["sources"].map(len) > 1).sum()),
        "items_parfumo_only": int((catalog["sources"].map(lambda s: list(s) == ["parfumo"])).sum()),
        "items_doevent_only": int((catalog["sources"].map(lambda s: list(s) == ["doevent"])).sum()),
        "brands": int(catalog["brand_key"].nunique()),
    }
    (out_dir / "index.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the catalog and embedding index.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=INDEX_DIR)
    arguments = parser.parse_args(argv)
    print(json.dumps(build_index(arguments.model_dir, arguments.out_dir), indent=2))


if __name__ == "__main__":
    main()
