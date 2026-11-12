import unittest

from tests import helpers
from scentvec.query import QueryParser, notes_matching, tokenize


class TokenizeTest(unittest.TestCase):
    def test_punctuation_and_minus(self):
        self.assertEqual(tokenize("Fresh citrus, no musk"), ["fresh", "citrus", ",", "no", "musk"])
        self.assertEqual(tokenize("vanilla -patchouli"), ["vanilla", "-", "patchouli"])


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser(helpers.NOTES + ["pink pepper"])

    def test_notes_and_exclusions(self):
        parsed = self.parser.parse("vanilla and tobacco, no patchouli")
        self.assertEqual(parsed.notes, ["vanilla", "tobacco"])
        self.assertEqual(parsed.avoid_notes, ["patchouli"])

    def test_negation_stops_at_but(self):
        parsed = self.parser.parse("without musk but with rose")
        self.assertEqual(parsed.avoid_notes, ["musk"])
        self.assertEqual(parsed.notes, ["rose"])

    def test_multi_word_note_and_alias(self):
        parsed = self.parser.parse("pink pepper with cedar")
        self.assertEqual(parsed.notes, ["pink pepper", "cedarwood"])

    def test_accords_gender_and_years(self):
        parsed = self.parser.parse("smoky woody scent for men from the 1990s")
        self.assertIn("Smoky", parsed.accords)
        self.assertIn("Woody", parsed.accords)
        self.assertEqual(parsed.genders, ["Male", "Unisex"])
        self.assertEqual((parsed.year_min, parsed.year_max), (1990, 1999))

    def test_unwanted_accord(self):
        parsed = self.parser.parse("rose, not sweet")
        self.assertEqual(parsed.notes, ["rose"])
        self.assertEqual(parsed.avoid_accords, ["Sweet"])

    def test_nonsense_is_reported_not_guessed(self):
        parsed = self.parser.parse("xyzzy plugh")
        self.assertTrue(parsed.is_empty())
        self.assertEqual(parsed.unrecognised, ["xyzzy", "plugh"])


class NoteMatchTest(unittest.TestCase):
    def test_whole_word_phrase_match(self):
        vocabulary = ["rose", "rosewood", "bulgarian rose", "pink pepper", "pepper"]
        self.assertEqual(notes_matching("rose", vocabulary), {"rose", "bulgarian rose"})
        self.assertEqual(notes_matching("pepper", vocabulary), {"pepper", "pink pepper"})


if __name__ == "__main__":
    unittest.main()
