import unittest

import numpy as np

from tests import helpers
from scentvec.encoder import canonical_pyramid
from scentvec.features import POSITION_BASE, POSITION_TOP, POSITION_UNKNOWN
from scentvec.metrics import best_relevant_ranks, retrieval_summary


class EncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = helpers.build_tiny_encoder()

    def test_vector_is_unit_length_with_the_model_dimension(self):
        vector = self.encoder.embed(top=["Bergamot"], base=["Musk", "Vanilla"])
        self.assertEqual(vector.shape, (self.encoder.dim,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_same_input_same_vector_alone_or_in_a_batch(self):
        perfume = canonical_pyramid(["bergamot"], ["rose"], ["musk"], [])
        other = canonical_pyramid(["tobacco", "vanilla", "amber", "saffron"], [], [], [])
        alone = self.encoder.encode([perfume])[0]
        batched = self.encoder.encode([other, perfume])[1]
        np.testing.assert_allclose(alone, batched, atol=1e-5)

    def test_raw_spellings_are_canonicalised(self):
        first = self.encoder.embed(notes=["Cedar", "Olibanum"])
        second = self.encoder.embed(notes=["cedarwood", "incense"])
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_no_notes_gives_the_zero_vector(self):
        self.assertEqual(float(np.abs(self.encoder.embed()).sum()), 0.0)

    def test_unknown_note_still_embeds(self):
        vector = self.encoder.embed(notes=["moon dust"])
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(self.encoder.known("moon dust"), "unknown")
        self.assertEqual(self.encoder.known("rose"), "note")
        self.assertEqual(self.encoder.known("wild rose"), "words")

    def test_head_scores_are_probabilities(self):
        scores = self.encoder.head_scores(self.encoder.embed(notes=["rose", "musk"]))
        self.assertEqual(scores["accords"].shape, (1, len(self.encoder.vocabulary.accords)))
        self.assertTrue(((scores["accords"] >= 0) & (scores["accords"] <= 1)).all())
        self.assertAlmostEqual(float(scores["family"].sum()), 1.0, places=5)


class PyramidTest(unittest.TestCase):
    def test_positions_and_duplicates(self):
        notes, positions = canonical_pyramid(["Lemon"], None, ["Musk", "lemon"], ["Rose"])
        self.assertEqual(notes, ["lemon", "musk", "rose"])
        self.assertEqual(positions, [POSITION_TOP, POSITION_BASE, POSITION_UNKNOWN])


class MetricsTest(unittest.TestCase):
    def test_ranks_and_summary(self):
        similarity = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.9], [0.3, 0.3, 0.3]])
        ranks = best_relevant_ranks(similarity, [[0], [1], [2]])
        # Third query: a three-way tie counts against the method.
        self.assertEqual(ranks.tolist(), [1, 2, 3])
        summary = retrieval_summary(ranks)
        self.assertAlmostEqual(summary["recall@1"], 1 / 3)
        self.assertAlmostEqual(summary["mrr"], (1 + 1 / 2 + 1 / 3) / 3)


if __name__ == "__main__":
    unittest.main()
