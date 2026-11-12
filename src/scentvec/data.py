"""Load the raw perfume databases into one clean table.

Sources (both are public snapshots; see README for licences):

* ``parfumo``  - TidyTuesday 2024-12-10 snapshot of Parfumo: note pyramid
  (top / middle / base), crowd-voted main accords, ratings.
* ``doevent``  - Hugging Face ``doevent/perfume``: flat ingredient list,
  expert olfactive family and subfamily, gender, description.

The two databases describe many of the same perfumes independently.  Those
overlaps are linked here because they are the only ground truth available
for "these two descriptions are the same smell".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .textnorm import (
    canonical_notes,
    detect_concentration,
    make_key,
    split_note_cell,
    strip_concentration,
)

# Data, runs, the model bundle and the index live under this directory.
# SCENTVEC_HOME overrides it for installs that are not run from the source tree.
PROJECT_ROOT = Path(os.environ.get("SCENTVEC_HOME") or Path(__file__).resolve().parents[2])
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PARFUMO_FILE = "parfumo_data_clean.csv"
DOEVENT_FILE = "doevent_perfumes.jsonl"

# Brand spellings that differ between the two databases (doevent -> parfumo).
BRAND_ALIASES = {
    "bulgari": "bvlgari",
    "jo malone london": "jo malone",
    "roja": "roja parfums",
    "l occitane": "l occitane en provence",
    "bond no 9 i love ny": "bond no 9",
    "rabanne": "paco rabanne",
    "afnan": "afnan perfumes",
    "memo": "memo paris",
    "nicolai": "nicolai parfums de nicolai",
    "donna karan dkny": "dkny donna karan",
    "demeter fragrance library": "demeter fragrance library the library of fragrance",
    "the library of fragrance": "demeter fragrance library the library of fragrance",
    "drops by toni cabal": "toni cabal drops",
    "pierre guillaume paris": "pierre guillaume",
    "dunhill london": "dunhill",
    "ungaro": "emanuel ungaro",
    "atkinsons london 1799": "atkinsons",
    "frederic malle": "editions de parfums frederic malle",
    "thoo the house of oud": "the house of oud",
    "nishane istanbul": "nishane",
    "memoize london": "memoize",
    "carner barcelona": "carner",
    "pacha ibiza": "pacha",
    "revlon": "revlon charles revson",
    "vilhelm": "vilhelm parfumerie",
    "victorinox swiss army": "victorinox",
    "mona di orio": "maison mona di orio",
    "armani prive": "giorgio armani",
    "emporio armani": "giorgio armani",
    "skinn": "skinn by titan",
    "adopt": "adopt reserve naturelle",
    "tfk": "the fragrance kitchen",
}

ACCORD_VOCABULARY = [
    "Floral", "Spicy", "Sweet", "Woody", "Fresh", "Fruity", "Citrus", "Green", "Powdery",
    "Synthetic", "Oriental", "Creamy", "Gourmand", "Resinous", "Aquatic", "Smoky", "Leathery",
    "Earthy", "Animal", "Chypre", "Fougère",
]

NOTE_COLUMNS = ("Top_Notes", "Middle_Notes", "Base_Notes")

# The Parfumo snapshot contains fabricated rows: invented note lists that end
# in a marker word ("Quarklox", "Sewer", "Burnt electronics", ...).  A marker
# is recognisable because it is almost always the *last* entry of its list,
# which no genuine note is (lists are mostly alphabetical; the most tail-heavy
# genuine notes reach about 0.8).
MARKER_MIN_COUNT = 20
MARKER_TAIL_SHARE = 0.9

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_SPACES = re.compile(r"\s+")


def _to_year(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = _YEAR.search(str(value))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1800 <= year <= 2030 else None


def _name_key(name: str, brand_key: str, year: Optional[int]) -> str:
    """Key for the perfume name with brand, year and concentration removed."""
    key = strip_concentration(make_key(name))
    if brand_key and key.startswith(brand_key + " ") and len(key) > len(brand_key) + 1:
        key = key[len(brand_key) + 1 :]
    if year is not None:
        key = re.sub(rf"\b{year}\b", " ", key)
    key = re.sub(r"\b(limited edition|ltd ed|edition)\b", " ", key)
    key = _SPACES.sub(" ", key).strip()
    return key or make_key(name)


_YEAR_PREFIX = re.compile(r"^(?:1[89]|20)\d{2}\b")
# A title cannot end in one of these: "Must de" + "Cartier" is one name, not name + brand.
_NAME_CONNECTORS = frozenset(
    "de du des d di da del della by of for pour par von the la le les et and & a à in on to "
    "from n +".split()
)


def _is_listing_suffix(tail: str, concentration: str) -> bool:
    """True when ``tail`` is what Parfumo appends after the brand: [year] [concentration]."""
    rest = _YEAR_PREFIX.sub("", tail.strip()).strip()
    if not rest:
        return True
    if concentration and rest.lower() == concentration.strip().lower():
        return True
    return strip_concentration(make_key(rest)) == ""


def _clean_parfumo_name(name: str, brand: str, concentration: str = "") -> str:
    """Remove the listing suffix Parfumo appends to the names of popular perfumes.

    'Light Blue Dolce & Gabbana 2001 Eau de Toilette' -> 'Light Blue'.  The brand
    is only cut where nothing but a year and a concentration follows it, so
    titles that contain the brand keep it:
    'Bleu de Chanel Chanel 2010 Eau de Toilette' -> 'Bleu de Chanel',
    'Miss Dior Eau Fraîche' -> unchanged.
    """
    name = name.strip()
    if not brand:
        return name
    marker = " " + brand
    start = name.find(marker)
    while start > 0:
        tail = name[start + len(marker) :]
        at_word_boundary = tail == "" or tail.startswith(" ")
        if at_word_boundary and _is_listing_suffix(tail, concentration):
            title = name[:start].strip()
            last_word = title.split()[-1].lower().strip("'’") if title else ""
            dangling = tail.strip() == "" and last_word in _NAME_CONNECTORS
            if title and not dangling:
                return title
        start = name.find(marker, start + 1)
    return name


def _raw_note_list(cell: object) -> List[str]:
    if not isinstance(cell, str):
        return []
    return [part.strip().lower() for part in cell.split(",") if part.strip()]


def find_marker_notes(
    note_lists: Iterable[List[str]],
    min_count: int = MARKER_MIN_COUNT,
    tail_share: float = MARKER_TAIL_SHARE,
) -> Set[str]:
    """Notes that are (almost) only ever the last entry of a multi-note list."""
    total: Counter = Counter()
    last: Counter = Counter()
    for notes in note_lists:
        total.update(notes)
        if len(notes) >= 2:
            last[notes[-1]] += 1
    return {
        note
        for note, count in total.items()
        if count >= min_count and last[note] / count >= tail_share
    }


def flag_fabricated_rows(frame: pd.DataFrame) -> Tuple[np.ndarray, Set[str]]:
    """Mark Parfumo rows that carry a marker note in any of their note lists.

    Such rows are dropped whole: where the other database describes the same
    perfume, their remaining notes agree with it no better than chance.
    """
    columns = {column: [_raw_note_list(cell) for cell in frame[column]] for column in NOTE_COLUMNS}
    markers = find_marker_notes(notes for column in NOTE_COLUMNS for notes in columns[column])
    flagged = np.zeros(len(frame), dtype=bool)
    for column in NOTE_COLUMNS:
        for position, notes in enumerate(columns[column]):
            if not flagged[position] and any(note in markers for note in notes):
                flagged[position] = True
    return flagged, markers


def load_parfumo(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Parfumo records.  ``result.attrs["quality"]`` reports what was dropped."""
    frame = pd.read_csv(
        raw_dir / PARFUMO_FILE, dtype=str, keep_default_na=False, na_values=["NA"]
    )
    # A few rows are stray table headers picked up by the original scraper.
    frame = frame[frame["Main_Accords"].fillna("") != "Main Accords"].reset_index(drop=True)
    fabricated, markers = flag_fabricated_rows(frame)

    records = []
    for row_number, row in enumerate(frame.itertuples(index=False)):
        if fabricated[row_number]:
            continue
        brand = str(row.Brand).strip()
        listed_concentration = row.Concentration if isinstance(row.Concentration, str) else ""
        display_name = _clean_parfumo_name(str(row.Name), brand, listed_concentration)
        year = _to_year(row.Release_Year)
        brand_key = make_key(brand)
        concentration = detect_concentration(row.Concentration) or detect_concentration(
            str(row.Name)[len(display_name) :]
        )
        accords = [
            accord for accord in split_note_cell(row.Main_Accords) if accord in ACCORD_VOCABULARY
        ]
        top = canonical_notes(split_note_cell(row.Top_Notes))
        middle = [n for n in canonical_notes(split_note_cell(row.Middle_Notes)) if n not in top]
        base = [
            n
            for n in canonical_notes(split_note_cell(row.Base_Notes))
            if n not in top and n not in middle
        ]
        rating = float(row.Rating_Value) if isinstance(row.Rating_Value, str) else np.nan
        rating_count = (
            int(float(row.Rating_Count)) if isinstance(row.Rating_Count, str) else 0
        )
        perfumers = (
            [part.strip() for part in str(row.Perfumers).split(",") if part.strip()]
            if isinstance(row.Perfumers, str)
            else []
        )
        records.append(
            {
                "record_id": f"p{row_number}",
                "source": "parfumo",
                "brand": brand,
                "name": display_name,
                "brand_key": brand_key,
                "name_key": _name_key(display_name, brand_key, None),
                "year": year,
                "concentration": concentration,
                "gender": None,
                "notes_top": top,
                "notes_middle": middle,
                "notes_base": base,
                "notes_flat": [],
                "accords": accords,
                "family": None,
                "subfamily": None,
                "rating": rating,
                "rating_count": rating_count,
                "perfumers": perfumers,
                "url": row.URL if isinstance(row.URL, str) else None,
                "description": None,
            }
        )
    result = pd.DataFrame.from_records(records)
    result.attrs["quality"] = {
        "parfumo_rows_read": int(len(frame)),
        "parfumo_fabricated_rows_dropped": int(fabricated.sum()),
        "parfumo_marker_notes": sorted(markers),
    }
    return result


def load_doevent(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    records = []
    with (raw_dir / DOEVENT_FILE).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            row = json.loads(line)
            brand = str(row.get("brand") or "").strip()
            name = str(row.get("name_perfume") or "").strip()
            year = _to_year(row.get("years"))
            raw_brand_key = make_key(brand)
            brand_key = BRAND_ALIASES.get(raw_brand_key, raw_brand_key)
            name_key = _name_key(name, raw_brand_key, year)
            family = str(row.get("family") or "").strip().title() or None
            subfamily = str(row.get("subfamily") or "").strip().title() or None
            description = row.get("description")
            records.append(
                {
                    "record_id": f"d{row_number}",
                    "source": "doevent",
                    "brand": brand,
                    "name": name,
                    "brand_key": brand_key,
                    "name_key": name_key,
                    "year": year,
                    "concentration": detect_concentration(name),
                    "gender": row.get("gender"),
                    "notes_top": [],
                    "notes_middle": [],
                    "notes_base": [],
                    "notes_flat": canonical_notes(row.get("ingredients") or []),
                    "accords": [],
                    "family": family,
                    "subfamily": subfamily,
                    "rating": np.nan,
                    "rating_count": 0,
                    "perfumers": [],
                    "url": None,
                    "description": description if isinstance(description, str) else None,
                }
            )
    return pd.DataFrame.from_records(records)


def all_notes(row) -> List[str]:
    """Every canonical note of a record, pyramid order first."""
    return (
        list(row["notes_top"])
        + list(row["notes_middle"])
        + list(row["notes_base"])
        + list(row["notes_flat"])
    )


def assign_split(brand_key: str, name_key: str) -> str:
    """Deterministic 80/10/10 split keyed on the perfume identity.

    Hashing (brand, name) keeps every description of one perfume - both
    databases, every concentration - on the same side of the split, so the
    test set never contains a perfume the model was trained on.
    """
    digest = hashlib.sha256(f"{brand_key}|{name_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def link_sources(records: pd.DataFrame) -> pd.DataFrame:
    """Pair each doevent record with the Parfumo records of the same perfume.

    A pair needs the same brand and name key, and release years within one
    year of each other when both are known.  ``n_candidates`` counts how many
    Parfumo records qualified; evaluation only uses pairs where it is 1.
    """
    with_notes = records[records["n_notes"] > 0]
    years = {
        position: (None if pd.isna(year) else int(year))
        for position, year in records["year"].items()
    }
    parfumo_index: Dict[tuple, List[int]] = defaultdict(list)
    for position, row in with_notes[with_notes["source"] == "parfumo"].iterrows():
        parfumo_index[(row["brand_key"], row["name_key"])].append(position)

    pairs = []
    for position, row in with_notes[with_notes["source"] == "doevent"].iterrows():
        candidates = parfumo_index.get((row["brand_key"], row["name_key"]), [])
        if not candidates:
            continue
        year = years[position]
        compatible = []
        for candidate in candidates:
            other_year = years[candidate]
            if year is not None and other_year is not None and abs(year - other_year) > 1:
                continue
            compatible.append(candidate)
        if row["concentration"]:
            same_concentration = [
                candidate
                for candidate in compatible
                if records.at[candidate, "concentration"] == row["concentration"]
            ]
            compatible = same_concentration or compatible
        for candidate in compatible:
            pairs.append(
                {
                    "doevent_id": row["record_id"],
                    "parfumo_id": records.at[candidate, "record_id"],
                    "n_candidates": len(compatible),
                    "split": row["split"],
                }
            )
    return pd.DataFrame.from_records(pairs)


def build_dataset(raw_dir: Path = RAW_DIR) -> Dict[str, pd.DataFrame]:
    parfumo = load_parfumo(raw_dir)
    quality = dict(parfumo.attrs.get("quality", {}))
    parfumo.attrs = {}
    records = pd.concat([parfumo, load_doevent(raw_dir)], ignore_index=True)
    records["year"] = records["year"].astype("Int64")
    records["n_notes"] = records.apply(lambda row: len(all_notes(row)), axis=1)
    records["split"] = [
        assign_split(brand_key, name_key)
        for brand_key, name_key in zip(records["brand_key"], records["name_key"])
    ]
    pairs = link_sources(records)
    return {"records": records, "pairs": pairs, "quality": quality}


def summarise(records: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    with_notes = records[records["n_notes"] > 0]
    note_counts: Counter = Counter()
    for row in with_notes.itertuples(index=False):
        note_counts.update(row.notes_top)
        note_counts.update(row.notes_middle)
        note_counts.update(row.notes_base)
        note_counts.update(row.notes_flat)
    unique_pairs = pairs[pairs["n_candidates"] == 1]
    return {
        "records_total": int(len(records)),
        "records_with_notes": int(len(with_notes)),
        "records_with_notes_by_source": with_notes["source"].value_counts().to_dict(),
        "records_with_notes_by_split": with_notes["split"].value_counts().to_dict(),
        "parfumo_with_notes_and_accords": int(
            ((with_notes["source"] == "parfumo") & (with_notes["accords"].map(len) > 0)).sum()
        ),
        "distinct_canonical_notes": int(len(note_counts)),
        "notes_seen_5_plus_times": int(sum(1 for count in note_counts.values() if count >= 5)),
        "cross_source_pairs": int(len(pairs)),
        "cross_source_linked_doevent_records": int(pairs["doevent_id"].nunique()),
        "cross_source_unique_pairs": int(len(unique_pairs)),
        "cross_source_unique_pairs_by_split": unique_pairs["split"].value_counts().to_dict(),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned perfume dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    arguments = parser.parse_args(argv)

    dataset = build_dataset(arguments.raw_dir)
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    dataset["records"].to_parquet(arguments.out_dir / "records.parquet", index=False)
    dataset["pairs"].to_parquet(arguments.out_dir / "pairs.parquet", index=False)
    summary = summarise(dataset["records"], dataset["pairs"])
    summary["quality"] = dataset["quality"]
    (arguments.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
