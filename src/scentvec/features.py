"""Turn note lists into the arrays the encoder consumes.

A note reaches the model through four channels:

* its own id, if the note is frequent enough in the training split;
* the ids of its words, so "burgundy blackcurrant" still lands next to
  "blackcurrant" even though the full string was never seen;
* an odor-chemistry profile (see :mod:`scentvec.chem`);
* its place in the pyramid: top, middle, base or unknown.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .chem import ChemTable
from .data import ACCORD_VOCABULARY
from .textnorm import note_tokens

PAD = 0
UNK = 1
MAX_TOKENS_PER_NOTE = 4
MAX_NOTES_PER_PERFUME = 40

POSITION_TOP = 0
POSITION_MIDDLE = 1
POSITION_BASE = 2
POSITION_UNKNOWN = 3
POSITION_NAMES = ["top", "middle", "base", "unknown"]

MIN_NOTE_COUNT = 5
MIN_TOKEN_COUNT = 3


@dataclass
class Vocabulary:
    notes: List[str]
    tokens: List[str]
    accords: List[str]
    families: List[str]
    subfamilies: List[str]

    def __post_init__(self) -> None:
        self.note_to_id: Dict[str, int] = {note: index + 2 for index, note in enumerate(self.notes)}
        self.token_to_id: Dict[str, int] = {
            token: index + 2 for index, token in enumerate(self.tokens)
        }
        self.accord_to_id = {accord: index for index, accord in enumerate(self.accords)}
        self.family_to_id = {family: index for index, family in enumerate(self.families)}
        self.subfamily_to_id = {
            subfamily: index for index, subfamily in enumerate(self.subfamilies)
        }

    @property
    def n_note_ids(self) -> int:
        return len(self.notes) + 2

    @property
    def n_token_ids(self) -> int:
        return len(self.tokens) + 2

    @classmethod
    def build(cls, train_records: pd.DataFrame) -> "Vocabulary":
        """Build from the training split only, so held-out notes stay unseen."""
        note_counts: Counter = Counter()
        token_counts: Counter = Counter()
        for column in ["notes_top", "notes_middle", "notes_base", "notes_flat"]:
            for notes in train_records[column]:
                for note in notes:
                    note_counts[note] += 1
                    token_counts.update(note_tokens(note))
        notes = sorted(note for note, count in note_counts.items() if count >= MIN_NOTE_COUNT)
        tokens = sorted(
            token for token, count in token_counts.items() if count >= MIN_TOKEN_COUNT
        )
        families = sorted(set(train_records["family"].dropna()))
        subfamilies = sorted(set(train_records["subfamily"].dropna()))
        return cls(notes, tokens, list(ACCORD_VOCABULARY), families, subfamilies)

    def to_json(self) -> dict:
        return {
            "notes": self.notes,
            "tokens": self.tokens,
            "accords": self.accords,
            "families": self.families,
            "subfamilies": self.subfamilies,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text())
        return cls(
            payload["notes"],
            payload["tokens"],
            payload["accords"],
            payload["families"],
            payload["subfamilies"],
        )


class NoteTable:
    """Per-note feature rows, shared by every perfume that uses the note."""

    def __init__(self, vocabulary: Vocabulary, chem_table: ChemTable) -> None:
        self.vocabulary = vocabulary
        self.chem_table = chem_table
        self.row_of: Dict[str, int] = {}
        self._note_ids: List[int] = []
        self._token_ids: List[List[int]] = []
        self._chem: List[np.ndarray] = []
        self._frozen: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def row(self, note: str) -> int:
        existing = self.row_of.get(note)
        if existing is not None:
            return existing
        vocabulary = self.vocabulary
        token_ids = [
            vocabulary.token_to_id.get(token, UNK)
            for token in note_tokens(note)[:MAX_TOKENS_PER_NOTE]
        ]
        token_ids += [PAD] * (MAX_TOKENS_PER_NOTE - len(token_ids))
        profile, _ = self.chem_table.lookup(note)
        if profile is None:
            profile = np.zeros(self.chem_table.dim, dtype=np.float32)
        self.row_of[note] = len(self._note_ids)
        self._note_ids.append(vocabulary.note_to_id.get(note, UNK))
        self._token_ids.append(token_ids)
        self._chem.append(profile)
        self._frozen = None
        return self.row_of[note]

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._frozen is None:
            self._frozen = (
                np.asarray(self._note_ids, dtype=np.int64),
                np.asarray(self._token_ids, dtype=np.int64).reshape(-1, MAX_TOKENS_PER_NOTE),
                np.asarray(self._chem, dtype=np.float32).reshape(-1, self.chem_table.dim),
            )
        return self._frozen


def record_notes_with_positions(row) -> Tuple[List[str], List[int]]:
    """Notes of a dataset record in pyramid order, with their position codes."""
    notes: List[str] = []
    positions: List[int] = []
    for column, position in [
        ("notes_top", POSITION_TOP),
        ("notes_middle", POSITION_MIDDLE),
        ("notes_base", POSITION_BASE),
        ("notes_flat", POSITION_UNKNOWN),
    ]:
        for note in row[column]:
            notes.append(note)
            positions.append(position)
    return notes, positions


def pad_batch(
    note_table: NoteTable,
    rows_per_item: Sequence[np.ndarray],
    positions_per_item: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Pad a batch of perfumes (each a list of note-table rows) to one length."""
    note_ids, token_ids, chem = note_table.arrays()
    batch_size = len(rows_per_item)
    max_length = max(1, max(len(rows) for rows in rows_per_item))
    batch_note_ids = np.zeros((batch_size, max_length), dtype=np.int64)
    batch_token_ids = np.zeros((batch_size, max_length, MAX_TOKENS_PER_NOTE), dtype=np.int64)
    batch_chem = np.zeros((batch_size, max_length, chem.shape[1]), dtype=np.float32)
    batch_positions = np.full((batch_size, max_length), POSITION_UNKNOWN, dtype=np.int64)
    batch_mask = np.zeros((batch_size, max_length), dtype=bool)
    for item, (rows, positions) in enumerate(zip(rows_per_item, positions_per_item)):
        length = len(rows)
        if length == 0:
            continue
        batch_note_ids[item, :length] = note_ids[rows]
        batch_token_ids[item, :length] = token_ids[rows]
        batch_chem[item, :length] = chem[rows]
        batch_positions[item, :length] = positions
        batch_mask[item, :length] = True
    return {
        "note_ids": batch_note_ids,
        "token_ids": batch_token_ids,
        "chem": batch_chem,
        "positions": batch_positions,
        "mask": batch_mask,
    }


class Corpus:
    """All records that have notes, pre-encoded for training and evaluation."""

    def __init__(self, records: pd.DataFrame, note_table: NoteTable) -> None:
        self.records = records.reset_index(drop=True)
        self.note_table = note_table
        vocabulary = note_table.vocabulary
        self.rows: List[np.ndarray] = []
        self.positions: List[np.ndarray] = []
        for _, row in self.records.iterrows():
            notes, positions = record_notes_with_positions(row)
            notes = notes[:MAX_NOTES_PER_PERFUME]
            positions = positions[:MAX_NOTES_PER_PERFUME]
            self.rows.append(np.asarray([note_table.row(note) for note in notes], dtype=np.int64))
            self.positions.append(np.asarray(positions, dtype=np.int64))

        count = len(self.records)
        self.accord_targets = np.zeros((count, len(vocabulary.accords)), dtype=np.float32)
        self.has_accords = np.zeros(count, dtype=bool)
        for index, accords in enumerate(self.records["accords"]):
            for rank, accord in enumerate(accords):
                self.accord_targets[index, vocabulary.accord_to_id[accord]] = accord_gain(rank)
            self.has_accords[index] = len(accords) > 0
        self.family = np.asarray(
            [vocabulary.family_to_id.get(value, -1) for value in self.records["family"]],
            dtype=np.int64,
        )
        self.subfamily = np.asarray(
            [vocabulary.subfamily_to_id.get(value, -1) for value in self.records["subfamily"]],
            dtype=np.int64,
        )
        self.index_of: Dict[str, int] = {
            record_id: index for index, record_id in enumerate(self.records["record_id"])
        }
        group_keys = self.records["brand_key"] + "|" + self.records["name_key"]
        self.group = pd.factorize(group_keys)[0].astype(np.int64)

    def batch(self, indices: Sequence[int]) -> Dict[str, np.ndarray]:
        return pad_batch(
            self.note_table,
            [self.rows[index] for index in indices],
            [self.positions[index] for index in indices],
        )


def accord_gain(rank: int) -> float:
    """Graded relevance of the accord voted into place ``rank`` (0 = strongest)."""
    return max(1.0 - 0.15 * rank, 0.3)


def load_corpus(
    processed_dir: Path, vocabulary: Optional[Vocabulary] = None
) -> Tuple[Corpus, pd.DataFrame, Vocabulary, ChemTable]:
    from .chem import CHEM_TABLE_FILE

    records = pd.read_parquet(processed_dir / "records.parquet")
    records = records[records["n_notes"] > 0].reset_index(drop=True)
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    chem_table = ChemTable.load(processed_dir / CHEM_TABLE_FILE)
    if vocabulary is None:
        vocabulary = Vocabulary.build(records[records["split"] == "train"])
    corpus = Corpus(records, NoteTable(vocabulary, chem_table))
    return corpus, pairs, vocabulary, chem_table
