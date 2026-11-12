import unittest

from tests import helpers  # noqa: F401 - puts src/ on the path
from scentvec.textnorm import (
    canonical_note,
    canonical_notes,
    detect_concentration,
    make_key,
    split_note_cell,
    strip_concentration,
)


class MakeKeyTest(unittest.TestCase):
    def test_folds_accents_case_and_punctuation(self):
        self.assertEqual(make_key("Fougère Royale!"), "fougere royale")
        self.assertEqual(make_key("Dolce & Gabbana"), "dolce and gabbana")

    def test_numero_sign(self):
        self.assertEqual(make_key("N°5"), "no 5")
        self.assertEqual(make_key("No. 5"), "no 5")


class CanonicalNoteTest(unittest.TestCase):
    def test_origin_and_parentheses_are_removed(self):
        self.assertEqual(canonical_note("Vetiver (Haiti)"), "vetiver")
        self.assertEqual(canonical_note("Clearwood™"), "clearwood")

    def test_botanical_qualifier_is_kept(self):
        self.assertEqual(canonical_note("Jasmine (Sambac)"), "jasmine sambac")

    def test_synonyms_collapse(self):
        self.assertEqual(canonical_note("Agarwood (Oud)"), "oud")
        self.assertEqual(canonical_note("Olibanum"), "incense")
        self.assertEqual(canonical_note("Cedar"), "cedarwood")
        self.assertEqual(canonical_note("Mandarin orange"), "mandarin")
        self.assertEqual(canonical_note("Green mandarin orange"), "green mandarin")

    def test_non_text_and_empty(self):
        self.assertIsNone(canonical_note(None))
        self.assertIsNone(canonical_note(float("nan")))
        self.assertIsNone(canonical_note("  ()  "))

    def test_list_drops_duplicates_in_order(self):
        self.assertEqual(
            canonical_notes(["Cedar", "Rose", "cedarwood", None, "Rose "]),
            ["cedarwood", "rose"],
        )


class ConcentrationTest(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(detect_concentration("Sauvage Eau de Parfum"), "edp")
        self.assertEqual(detect_concentration("Extrait de Parfum"), "extrait")
        self.assertEqual(detect_concentration("After Shave Lotion"), "aftershave")
        self.assertIsNone(detect_concentration("Sauvage"))
        self.assertIsNone(detect_concentration(None))

    def test_strip(self):
        self.assertEqual(strip_concentration("sauvage eau de toilette"), "sauvage")


class SplitCellTest(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_note_cell("Rose,  Musk , ,Amber"), ["Rose", "Musk", "Amber"])
        self.assertEqual(split_note_cell(float("nan")), [])


if __name__ == "__main__":
    unittest.main()
