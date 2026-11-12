"""Inference wrapper: raw note strings in, embedding out.

This is the "embedding model" in the narrow sense.  It needs only the model
bundle (weights, vocabulary, chemistry table) and no perfume database.

    encoder = NoteEncoder.load("artifacts/model")
    vector = encoder.embed(top=["bergamot"], middle=["rose"], base=["musk"])
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .chem import CHEM_TABLE_FILE, ChemTable
from .features import (
    MAX_NOTES_PER_PERFUME,
    POSITION_BASE,
    POSITION_MIDDLE,
    POSITION_TOP,
    POSITION_UNKNOWN,
    NoteTable,
    Vocabulary,
    pad_batch,
)
from .model import ModelConfig, ScentEncoder, to_tensors
from .textnorm import canonical_notes

NotesWithPositions = Tuple[List[str], List[int]]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def canonical_pyramid(
    top: Optional[Iterable[str]] = None,
    middle: Optional[Iterable[str]] = None,
    base: Optional[Iterable[str]] = None,
    notes: Optional[Iterable[str]] = None,
) -> NotesWithPositions:
    """Canonicalise raw note strings; ``notes`` are notes of unknown position."""
    result_notes: List[str] = []
    result_positions: List[int] = []
    for raw, position in [
        (top, POSITION_TOP),
        (middle, POSITION_MIDDLE),
        (base, POSITION_BASE),
        (notes, POSITION_UNKNOWN),
    ]:
        for note in canonical_notes(raw or []):
            if note not in result_notes:
                result_notes.append(note)
                result_positions.append(position)
    return result_notes[:MAX_NOTES_PER_PERFUME], result_positions[:MAX_NOTES_PER_PERFUME]


class NoteEncoder:
    def __init__(self, model: ScentEncoder, vocabulary: Vocabulary, chem_table: ChemTable) -> None:
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.chem_table = chem_table
        self._note_table = NoteTable(vocabulary, chem_table)
        self._lock = threading.Lock()
        self.accord_prototypes = model.accord_head.unit_prototypes().numpy()
        self.family_prototypes = model.family_head.unit_prototypes().numpy()
        self.subfamily_prototypes = model.subfamily_head.unit_prototypes().numpy()

    @property
    def dim(self) -> int:
        return self.model.config.embed_dim

    @classmethod
    def load(cls, model_dir: Path) -> "NoteEncoder":
        model_dir = Path(model_dir)
        saved = json.loads((model_dir / "config.json").read_text())
        model = ScentEncoder(ModelConfig(**saved["model"]))
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        vocabulary = Vocabulary.load(model_dir / "vocab.json")
        chem_table = ChemTable.load(model_dir / CHEM_TABLE_FILE)
        return cls(model, vocabulary, chem_table)

    def encode(self, perfumes: Sequence[NotesWithPositions], batch_size: int = 1024) -> np.ndarray:
        """Embed perfumes given as (canonical notes, position codes).

        A perfume with no notes gets the zero vector.
        """
        output = np.zeros((len(perfumes), self.dim), dtype=np.float32)
        with self._lock:
            encoded = [
                (
                    np.asarray([self._note_table.row(note) for note in notes], dtype=np.int64),
                    np.asarray(positions, dtype=np.int64),
                )
                for notes, positions in perfumes
            ]
            with torch.no_grad():
                for start in range(0, len(encoded), batch_size):
                    chunk = encoded[start : start + batch_size]
                    batch = to_tensors(
                        pad_batch(
                            self._note_table,
                            [rows for rows, _ in chunk],
                            [positions for _, positions in chunk],
                        )
                    )
                    vectors = self.model(**batch).numpy()
                    for offset, (rows, _) in enumerate(chunk):
                        if len(rows) > 0:
                            output[start + offset] = vectors[offset]
        return output

    def embed(
        self,
        top: Optional[Iterable[str]] = None,
        middle: Optional[Iterable[str]] = None,
        base: Optional[Iterable[str]] = None,
        notes: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        """Embed one perfume from raw note strings."""
        return self.encode([canonical_pyramid(top, middle, base, notes)])[0]

    def known(self, note: str) -> str:
        """How the model sees a canonical note: 'note', 'words', 'chemistry' or 'unknown'."""
        if note in self.vocabulary.note_to_id:
            return "note"
        if any(token in self.vocabulary.token_to_id for token in note.split()):
            return "words"
        if self.chem_table.lookup(note)[0] is not None:
            return "chemistry"
        return "unknown"

    def head_scores(self, vectors: np.ndarray) -> Dict[str, np.ndarray]:
        """Accord probabilities and family / subfamily distributions for embeddings."""
        with torch.no_grad():
            heads = self.model.heads(torch.from_numpy(np.atleast_2d(vectors).astype(np.float32)))
        return {
            "accords": _sigmoid(heads["accords"].numpy()),
            "family": _softmax(heads["family"].numpy()),
            "subfamily": _softmax(heads["subfamily"].numpy()),
        }
