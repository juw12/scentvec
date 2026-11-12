import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests import helpers  # noqa: F401 - puts src/ on the path
from scentvec import data


def _parfumo_frame(rows):
    columns = [
        "Number", "Name", "Brand", "Release_Year", "Concentration", "Rating_Value",
        "Rating_Count", "Main_Accords", "Top_Notes", "Middle_Notes", "Base_Notes",
        "Perfumers", "URL",
    ]
    return pd.DataFrame(rows, columns=columns)


def _genuine_row(index):
    # Alphabetical lists, like the real database: no note is always last.
    tops = ["Bergamot, Lemon", "Lemon, Rose", "Bergamot, Rose", "Rose, Bergamot"]
    return [
        str(index), f"Scent {index}", "House", "2001", "Eau de Parfum", "7.5", "120",
        "Fresh, Citrus", tops[index % 4], "Jasmine, Rose", "Amber, Musk", "A. Nose",
        f"https://example.invalid/{index}",
    ]


def _fabricated_row(index):
    return [
        str(index), f"Fake {index}", "House", "2002", "", "", "",
        "", "Lemon, Rose, Quarklox", "Jasmine", "Musk, Amber", "", "",
    ]


class MarkerNoteTest(unittest.TestCase):
    def test_note_that_is_always_last_is_a_marker(self):
        lists = [["lemon", "rose", "quarklox"]] * 25 + [["bergamot", "lemon"], ["lemon", "rose"]] * 30
        self.assertEqual(data.find_marker_notes(lists), {"quarklox"})

    def test_rare_tail_note_is_not_flagged(self):
        lists = [["lemon", "wild jasmine"]] * 10 + [["bergamot", "lemon"], ["lemon", "bergamot"]] * 30
        self.assertEqual(data.find_marker_notes(lists), set())

    def test_genuine_alphabetical_tail_is_not_flagged(self):
        # "ylang" sorts last often, but not nine times out of ten.
        lists = [["rose", "ylang"]] * 24 + [["ylang", "zest"]] * 6 + [["amber", "rose"]] * 30
        self.assertNotIn("ylang", data.find_marker_notes(lists))


class LoadParfumoTest(unittest.TestCase):
    def setUp(self):
        rows = [_genuine_row(index) for index in range(60)]
        rows += [_fabricated_row(index) for index in range(60, 85)]
        self.frame = _parfumo_frame(rows)

    def test_fabricated_rows_are_flagged(self):
        flagged, markers = data.flag_fabricated_rows(self.frame)
        self.assertEqual(markers, {"quarklox"})
        self.assertEqual(int(flagged.sum()), 25)
        self.assertFalse(flagged[:60].any())

    def test_loader_drops_them_and_keeps_row_ids_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.frame.to_csv(Path(folder) / data.PARFUMO_FILE, index=False)
            records = data.load_parfumo(Path(folder))
        self.assertEqual(len(records), 60)
        self.assertEqual(records.attrs["quality"]["parfumo_fabricated_rows_dropped"], 25)
        self.assertEqual(list(records["record_id"])[:3], ["p0", "p1", "p2"])
        every_note = {note for notes in records["notes_top"] for note in notes}
        self.assertNotIn("quarklox", every_note)
        first = records.iloc[0]
        self.assertEqual(first["concentration"], "edp")
        self.assertEqual(first["year"], 2001)
        self.assertEqual(first["accords"], ["Fresh", "Citrus"])
        self.assertEqual(first["notes_base"], ["amber", "musk"])


class CleanParfumoNameTest(unittest.TestCase):
    def test_listing_suffix_is_removed(self):
        cleaned = data._clean_parfumo_name(
            "Light Blue Dolce & Gabbana 2001 Eau de Toilette",
            "Dolce & Gabbana",
            "Eau de Toilette",
        )
        self.assertEqual(cleaned, "Light Blue")

    def test_brand_inside_the_title_is_kept(self):
        cases = [
            ("Bleu de Chanel Chanel 2010 Eau de Toilette", "Chanel", "Eau de Toilette", "Bleu de Chanel"),
            ("Gucci by Gucci Gucci 2008 Eau de Parfum", "Gucci", "Eau de Parfum", "Gucci by Gucci"),
            ("Miss Dior Eau Fraîche", "Dior", "", "Miss Dior Eau Fraîche"),
            ("L'Eau de Missha - Always Be Mine", "Missha", "", "L'Eau de Missha - Always Be Mine"),
            ("Must de Cartier", "Cartier", "", "Must de Cartier"),
        ]
        for name, brand, concentration, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(data._clean_parfumo_name(name, brand, concentration), expected)

    def test_brand_with_punctuation_and_no_suffix(self):
        self.assertEqual(data._clean_parfumo_name("Fruit Kelly + Jones", "Kelly + Jones"), "Fruit")

    def test_missing_brand_leaves_the_name_alone(self):
        self.assertEqual(data._clean_parfumo_name("  Something  ", ""), "Something")

    def test_name_equal_to_brand_is_not_emptied(self):
        self.assertEqual(data._clean_parfumo_name("Chanel", "Chanel"), "Chanel")


class SplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_roughly_80_10_10(self):
        self.assertEqual(data.assign_split("creed", "aventus"), data.assign_split("creed", "aventus"))
        counts = {"train": 0, "val": 0, "test": 0}
        for index in range(5000):
            counts[data.assign_split("brand", f"name {index}")] += 1
        self.assertAlmostEqual(counts["train"] / 5000, 0.8, delta=0.03)
        self.assertAlmostEqual(counts["val"] / 5000, 0.1, delta=0.02)
        self.assertAlmostEqual(counts["test"] / 5000, 0.1, delta=0.02)


class LinkTest(unittest.TestCase):
    def _records(self):
        rows = [
            ("p0", "parfumo", "creed", "aventus", 2010, "edp"),
            ("p1", "parfumo", "creed", "aventus", 1990, None),
            ("p2", "parfumo", "creed", "himalaya", 2002, None),
            ("d0", "doevent", "creed", "aventus", 2010, None),
            ("d1", "doevent", "creed", "erolfa", 1992, None),
        ]
        records = pd.DataFrame(
            rows, columns=["record_id", "source", "brand_key", "name_key", "year", "concentration"]
        )
        records["year"] = records["year"].astype("Int64")
        records["n_notes"] = 3
        records["split"] = [
            data.assign_split(brand, name)
            for brand, name in zip(records["brand_key"], records["name_key"])
        ]
        return records

    def test_same_perfume_is_linked_and_year_must_agree(self):
        records = self._records()
        pairs = data.link_sources(records)
        self.assertEqual(pairs[["doevent_id", "parfumo_id"]].values.tolist(), [["d0", "p0"]])
        self.assertEqual(int(pairs.iloc[0]["n_candidates"]), 1)

    def test_both_descriptions_share_a_split(self):
        records = self._records().set_index("record_id")
        self.assertEqual(records.loc["p0", "split"], records.loc["d0", "split"])


if __name__ == "__main__":
    unittest.main()
