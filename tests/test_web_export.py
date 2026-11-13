"""The browser export in docs/ must stay in step with the packaged model."""

import hashlib
import json
import re
import unittest
from pathlib import Path

import numpy as np

from tests import helpers  # noqa: F401  (puts src/ on the path)
from scentvec.encoder import NoteEncoder

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODEL_DIR = ROOT / "artifacts" / "model"
POSITION_KEYS = ("top", "middle", "base", "notes")


def page_presets():
    """Note lists of the example formulas offered by the landing page."""
    page = (DOCS / "index.html").read_text(encoding="utf-8")
    block = re.search(r"const PRESETS = \[(.*?)\n  \];", page, flags=re.S).group(1)
    presets = []
    for line in block.splitlines():
        name = re.search(r'name: "([^"]+)"', line)
        if not name:
            continue
        notes = []
        for position in ("top", "middle", "base"):
            listed = re.search(position + r": \[([^\]]*)\]", line).group(1)
            notes.extend(re.findall(r'"([^"]+)"', listed))
        presets.append((name.group(1), notes))
    return presets


class WebExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DOCS / "model.json").read_text(encoding="utf-8"))
        cls.binary = (DOCS / cls.manifest["binary"]).read_bytes()

    def test_binary_matches_the_manifest(self):
        self.assertEqual(len(self.binary), self.manifest["binary_bytes"])
        self.assertEqual(hashlib.sha256(self.binary).hexdigest(), self.manifest["binary_sha256"])

    def test_tensors_tile_the_binary_without_gaps_or_overlap(self):
        tensors = sorted(self.manifest["tensors"].values(), key=lambda entry: entry["offset"])
        expected_offset = 0
        for entry in tensors:
            self.assertEqual(entry["offset"], expected_offset)
            self.assertEqual(entry["length"], int(np.prod(entry["shape"])))
            expected_offset += entry["length"]
        self.assertEqual(expected_offset * 2, len(self.binary))

    def test_weights_are_finite(self):
        values = np.frombuffer(self.binary, dtype="<f2")
        self.assertTrue(np.isfinite(values).all())

    def test_vocabulary_is_unique_and_matches_the_note_table(self):
        notes = self.manifest["notes"]
        self.assertEqual(len(notes), len(set(notes)))
        self.assertEqual(self.manifest["tensors"]["note_vectors"]["shape"][0], len(notes))

    def test_page_examples_only_use_notes_the_export_knows(self):
        known = set(self.manifest["notes"])
        presets = page_presets()
        self.assertGreaterEqual(len(presets), 2)
        for name, notes in presets:
            self.assertTrue(notes, name)
            self.assertEqual([note for note in notes if note not in known], [], name)

    def test_reference_pyramids_use_known_notes_and_unit_vectors(self):
        known = set(self.manifest["notes"])
        self.assertGreaterEqual(len(self.manifest["reference"]), 3)
        for reference in self.manifest["reference"]:
            for position, notes in reference["pyramid"].items():
                self.assertIn(position, POSITION_KEYS)
                self.assertEqual([note for note in notes if note not in known], [])
            norm = float(np.linalg.norm(reference["embedding"]))
            self.assertAlmostEqual(norm, 1.0, places=3)

    @unittest.skipUnless((MODEL_DIR / "model.pt").exists(), "packaged model not present")
    def test_reference_embeddings_match_the_packaged_model(self):
        encoder = NoteEncoder.load(MODEL_DIR)
        self.assertEqual(self.manifest["embed_dim"], encoder.dim)
        self.assertEqual(self.manifest["accords"], list(encoder.vocabulary.accords))
        for reference in self.manifest["reference"]:
            expected = np.asarray(reference["embedding"], dtype=np.float64)
            actual = encoder.embed(**reference["pyramid"]).astype(np.float64)
            cosine = float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))
            self.assertGreater(cosine, 0.9999)


if __name__ == "__main__":
    unittest.main()
