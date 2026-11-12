"""Engine behaviour on the tiny hermetic catalog."""

from __future__ import annotations

import unittest

import numpy as np

from tests import helpers
from scentvec.engine import EngineError, Filters

AVENTUS, AVENTUS_HER, AVENTUS_COLOGNE, CLUB_DE_NUIT, BACCARAT, TOBACCO_VANILLE, CHANEL_5, QUIET_ROSE = range(8)


def ids(results):
    return [entry["item_id"] for entry in results]


class EngineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = helpers.build_tiny_engine()


class LookupTest(EngineTestCase):
    def test_brand_and_name_prefers_the_plain_perfume(self):
        results = self.engine.lookup("creed aventus")
        self.assertEqual(results[0]["item_id"], AVENTUS)
        self.assertTrue(results[0]["matched_all_words"])
        self.assertEqual(set(ids(results)), {AVENTUS, AVENTUS_HER, AVENTUS_COLOGNE})

    def test_last_word_may_be_a_prefix(self):
        self.assertEqual(ids(self.engine.lookup("baccarat rou")), [BACCARAT])

    def test_accents_and_symbols_do_not_matter(self):
        self.assertEqual(self.engine.lookup("chanel n5")[0]["item_id"], CHANEL_5)

    def test_no_match_and_empty_query(self):
        self.assertEqual(self.engine.lookup("zzzz qqqq"), [])
        self.assertEqual(self.engine.lookup("   "), [])

    def test_limit(self):
        self.assertEqual(len(self.engine.lookup("creed", limit=2)), 2)


class SimilarTest(EngineTestCase):
    def test_same_line_is_excluded_by_default(self):
        found = ids(self.engine.similar(AVENTUS, k=7)["results"])
        self.assertNotIn(AVENTUS, found)
        self.assertNotIn(AVENTUS_HER, found)
        self.assertNotIn(AVENTUS_COLOGNE, found)
        self.assertIn(CLUB_DE_NUIT, found)

    def test_same_line_can_be_kept(self):
        found = ids(self.engine.similar(AVENTUS, k=7, exclude_same_line=False)["results"])
        self.assertNotIn(AVENTUS, found)
        self.assertIn(AVENTUS_HER, found)
        self.assertIn(AVENTUS_COLOGNE, found)

    def test_same_line_mask_is_symmetric_and_brand_bound(self):
        mask = self.engine.same_line_mask(AVENTUS_COLOGNE)
        self.assertTrue(mask[AVENTUS])
        self.assertFalse(mask[AVENTUS_HER])
        self.assertFalse(mask[CLUB_DE_NUIT])

    def test_scores_are_sorted_cosines(self):
        results = self.engine.similar(BACCARAT, k=5)["results"]
        scores = [entry["score"] for entry in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(-1.0001 <= score <= 1.0001 for score in scores))

    def test_shared_notes_are_reported(self):
        results = self.engine.similar(AVENTUS, k=7)["results"]
        club = next(entry for entry in results if entry["item_id"] == CLUB_DE_NUIT)
        self.assertIn("pineapple", club["shared_notes"])
        self.assertIn("birch", club["shared_notes"])

    def test_filters_apply(self):
        filters = Filters.from_json({"genders": ["Female"]})
        found = ids(self.engine.similar(AVENTUS, k=7, filters=filters)["results"])
        self.assertEqual(found, [CHANEL_5])

    def test_unknown_and_bad_ids(self):
        for bad in (99, -1, "0", 1.5, True, None):
            with self.assertRaises(EngineError):
                self.engine.similar(bad)


class FiltersTest(EngineTestCase):
    def test_aliases_and_single_strings(self):
        filters = Filters.from_json({"gender": "Male", "must": ["rose"], "min_ratings": 10})
        self.assertEqual(filters.genders, ["Male"])
        self.assertEqual(filters.must_notes, ["rose"])
        self.assertEqual(filters.min_rating_count, 10)

    def test_empty(self):
        self.assertEqual(Filters.from_json(None), Filters())
        self.assertEqual(Filters.from_json({}), Filters())

    def test_rejects_bad_input(self):
        bad_payloads = [
            {"nonsense": 1},
            {"genders": 5},
            {"genders": ["Male", 3]},
            {"year_min": "soon"},
            {"year_min": 1990.5},
            {"min_rating": float("nan")},
            {"min_rating_count": True},
            {"include_unknown_gender": "yes"},
            {"gender": "Male", "genders": ["Female"]},
            {"brands": ["x"] * 51},
            ["genders"],
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(EngineError):
                    Filters.from_json(payload)

    def test_year_brand_and_rating_count(self):
        engine = self.engine
        recent = engine.filter_mask(Filters.from_json({"year_min": 2015}))
        self.assertEqual(set(np.flatnonzero(recent)), {AVENTUS_HER, AVENTUS_COLOGNE, CLUB_DE_NUIT, BACCARAT})
        creed = engine.filter_mask(Filters.from_json({"brand": "Creed"}))
        self.assertEqual(set(np.flatnonzero(creed)), {AVENTUS, AVENTUS_HER, AVENTUS_COLOGNE})
        not_creed = engine.filter_mask(Filters.from_json({"exclude_brand": "creed"}))
        self.assertFalse(not_creed[AVENTUS])
        popular = engine.filter_mask(Filters.from_json({"min_rating_count": 1000}))
        self.assertFalse(popular[QUIET_ROSE])
        self.assertFalse(popular[AVENTUS_COLOGNE])
        self.assertTrue(popular[AVENTUS])

    def test_must_and_exclude_notes(self):
        mask = self.engine.filter_mask(Filters.from_json({"must": ["rose"], "avoid": ["lemon"]}))
        self.assertEqual(set(np.flatnonzero(mask)), {QUIET_ROSE})


class SearchTest(EngineTestCase):
    def test_notes_rank_literal_matches_first(self):
        result = self.engine.search(notes=["tobacco", "vanilla"], k=3)
        self.assertEqual(result["results"][0]["item_id"], TOBACCO_VANILLE)
        self.assertEqual(result["results"][0]["matched_notes"], ["tobacco", "vanilla"])
        self.assertEqual(result["results"][0]["score_parts"]["note_match"], 1.0)

    def test_avoid_notes_remove_perfumes(self):
        result = self.engine.search(notes=["rose"], avoid_notes=["musk"], k=8)
        self.assertEqual(result["query"]["avoid_notes"], ["musk"])
        for entry in result["results"]:
            listed = entry["notes_top"] + entry["notes_middle"] + entry["notes_base"]
            self.assertNotIn("musk", listed)

    def test_text_query(self):
        # The parser only knows notes carried by at least three catalog perfumes.
        result = self.engine.search(text="amber and vanilla without rose", k=8)
        self.assertEqual([note["note"] for note in result["query"]["notes"]], ["amber", "vanilla"])
        self.assertEqual(result["query"]["avoid_notes"], ["rose"])
        by_id = {entry["item_id"]: entry for entry in result["results"]}
        self.assertEqual(by_id[TOBACCO_VANILLE]["matched_notes"], ["amber", "vanilla"])
        self.assertTrue({AVENTUS_HER, CLUB_DE_NUIT, CHANEL_5, QUIET_ROSE}.isdisjoint(by_id))

    def test_rare_words_are_reported_as_unrecognised(self):
        result = self.engine.search(text="vanilla with tobacco", k=3)
        self.assertEqual(result["query"]["unrecognised"], ["tobacco"])

    def test_text_reference_excludes_the_reference_line(self):
        result = self.engine.search(text="like creed aventus", k=8)
        self.assertEqual(ids(result["query"]["like"]), [AVENTUS])
        found = ids(result["results"])
        self.assertTrue({AVENTUS, AVENTUS_HER, AVENTUS_COLOGNE}.isdisjoint(found))

    def test_blend_matches_manual_computation(self):
        engine = self.engine
        result = engine.search(like=[BACCARAT, TOBACCO_VANILLE], like_weights=[3, 1], k=8)
        blend = 0.75 * engine.embeddings[BACCARAT] + 0.25 * engine.embeddings[TOBACCO_VANILLE]
        blend /= np.linalg.norm(blend)
        expected = engine.embeddings @ blend
        found = ids(result["results"])
        self.assertTrue({BACCARAT, TOBACCO_VANILLE}.isdisjoint(found))
        for entry in result["results"]:
            self.assertAlmostEqual(entry["score_parts"]["similarity"], float(expected[entry["item_id"]]), places=3)

    def test_blend_weight_errors(self):
        for weights in ([1], [1, -1], [0, 0], "11", [1, "x"], [1, float("inf")]):
            with self.subTest(weights=weights):
                with self.assertRaises(EngineError):
                    self.engine.search(like=[BACCARAT, TOBACCO_VANILLE], like_weights=weights)

    def test_nothing_to_search(self):
        with self.assertRaises(EngineError):
            self.engine.search()
        with self.assertRaises(EngineError):
            self.engine.search(text="xyzzy plugh")

    def test_bad_arguments(self):
        with self.assertRaises(EngineError):
            self.engine.search(notes=[1, 2])
        with self.assertRaises(EngineError):
            self.engine.search(accords=["not an accord"])
        with self.assertRaises(EngineError):
            self.engine.search(accords=7)
        with self.assertRaises(EngineError):
            self.engine.search(like=[99])

    def test_filters_are_reported_and_applied(self):
        result = self.engine.search(notes=["jasmine"], filters=Filters.from_json({"gender": "Unisex"}), k=8)
        self.assertEqual(result["query"]["filters"], {"genders": ["Unisex"]})
        self.assertEqual(result["candidates"], 2)
        self.assertEqual(set(ids(result["results"])), {BACCARAT, TOBACCO_VANILLE})


class DescribeAndMiscTest(EngineTestCase):
    def test_describe(self):
        described = self.engine.describe(top=["Bergamot"], base=["vanilla", "not-a-real-note"])
        self.assertEqual(described["dim"], 16)
        self.assertEqual(len(described["vector"]), 16)
        self.assertAlmostEqual(float(np.linalg.norm(described["vector"])), 1.0, places=4)
        self.assertEqual(described["notes"][0]["position"], "top")
        self.assertEqual(len(described["family"]), 3)
        without_vector = self.engine.describe(notes=["rose"], include_vector=False)
        self.assertNotIn("vector", without_vector)

    def test_describe_needs_a_note(self):
        with self.assertRaises(EngineError):
            self.engine.describe()

    def test_explain(self):
        explanation = self.engine.explain(AVENTUS, CLUB_DE_NUIT)
        self.assertIn("pineapple", explanation["shared_notes"])
        self.assertNotIn("rose", explanation["shared_notes"])

    def test_suggest_notes(self):
        suggestions = self.engine.suggest_notes("ro")
        self.assertEqual(suggestions[0]["note"], "rose")
        self.assertEqual(suggestions[0]["perfumes"], 4)
        self.assertEqual(self.engine.suggest_notes(""), [])

    def test_stats_and_get(self):
        stats = self.engine.stats()
        self.assertEqual(stats["perfumes"], 8)
        self.assertEqual(stats["embedding_dim"], 16)
        self.assertEqual(self.engine.get(CHANEL_5)["name"], "N°5")
        self.assertEqual(len(self.engine), 8)


if __name__ == "__main__":
    unittest.main()
