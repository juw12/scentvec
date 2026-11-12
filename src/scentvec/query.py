"""Parse a free-text scent request into structured search terms.

    "like aventus but sweeter, no pineapple, for men after 2015"

becomes a reference perfume, a wanted accord, an excluded note, a gender
filter and a year filter.  The parser is a plain dictionary matcher: it only
understands note names, accord adjectives, a few filter phrases and
negation.  Words it does not understand are returned to the caller instead
of being silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .textnorm import NOTE_ALIASES, ascii_fold

# Adjectives (and their comparatives) that name one of the 21 voted accords.
ACCORD_WORDS: Dict[str, str] = {
    "floral": "Floral",
    "flowery": "Floral",
    "spicy": "Spicy",
    "spicier": "Spicy",
    "sweet": "Sweet",
    "sweeter": "Sweet",
    "sugary": "Sweet",
    "woody": "Woody",
    "woodier": "Woody",
    "woodsy": "Woody",
    "fresh": "Fresh",
    "fresher": "Fresh",
    "clean": "Fresh",
    "cleaner": "Fresh",
    "crisp": "Fresh",
    "fruity": "Fruity",
    "fruitier": "Fruity",
    "citrus": "Citrus",
    "citrusy": "Citrus",
    "citric": "Citrus",
    "zesty": "Citrus",
    "green": "Green",
    "greener": "Green",
    "powdery": "Powdery",
    "powderier": "Powdery",
    "synthetic": "Synthetic",
    "chemical": "Synthetic",
    "oriental": "Oriental",
    "ambery": "Oriental",
    "creamy": "Creamy",
    "creamier": "Creamy",
    "gourmand": "Gourmand",
    "edible": "Gourmand",
    "dessert": "Gourmand",
    "resinous": "Resinous",
    "balsamic": "Resinous",
    "aquatic": "Aquatic",
    "watery": "Aquatic",
    "oceanic": "Aquatic",
    "smoky": "Smoky",
    "smokey": "Smoky",
    "smokier": "Smoky",
    "leathery": "Leathery",
    "earthy": "Earthy",
    "earthier": "Earthy",
    "animalic": "Animal",
    "animal": "Animal",
    "skanky": "Animal",
    "chypre": "Chypre",
    "fougere": "Fougère",
    "barbershop": "Fougère",
}

NEGATORS = frozenset(
    [
        "no", "not", "without", "non", "minus", "avoid", "less", "never", "hate", "exclude",
        "excluding", "zero", "dont", "doesnt", "isnt", "nothing", "-",
    ]
)
SCOPE_RESETS = frozenset(["but", "with", "plus", "more", "yet", "however"])
SEPARATORS = frozenset([",", ";", "."])
REFERENCE_TRIGGERS: Dict[Tuple[str, ...], bool] = {
    # value: True when the phrase can only introduce a perfume name
    ("similar", "to"): True,
    ("alternative", "to"): True,
    ("alternatives", "to"): True,
    ("dupe", "of"): True,
    ("dupes", "of"): True,
    ("dupe", "for"): True,
    ("clone", "of"): True,
    ("clones", "of"): True,
    ("inspired", "by"): True,
    ("smells", "like"): False,
    ("like",): False,
}
MALE_WORDS = frozenset(["men", "mens", "man", "masculine", "male", "him", "guys"])
FEMALE_WORDS = frozenset(["women", "womens", "woman", "feminine", "female", "her", "ladies"])
STOP_WORDS = frozenset(
    """
    a an the and or of to i im want something scent scents perfume perfumes fragrance fragrances
    smell smells smelling that is are be very too bit little slightly note notes some lots lot
    in on it for me my please find looking need kind sort really quite accord accords vibe vibes
    type one ones any which what with touch hint hints of dry down drydown opening heart base top
    middle s t than want wants prefer would should have has do does like love
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+|[,;.]")
_LEADING_MINUS = re.compile(r"(^|\s)-(?=[a-z])")
_YEAR = re.compile(r"^(1[89]\d\d|20\d\d)$")
_DECADE = re.compile(r"^(1[89]\d0|20\d0)s$")

MAX_PHRASE_WORDS = 4
MAX_REFERENCE_WORDS = 8


@dataclass
class ParsedQuery:
    notes: List[str] = field(default_factory=list)
    accords: List[str] = field(default_factory=list)
    avoid_notes: List[str] = field(default_factory=list)
    avoid_accords: List[str] = field(default_factory=list)
    like: List[int] = field(default_factory=list)
    unlike: List[int] = field(default_factory=list)
    genders: List[str] = field(default_factory=list)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    unrecognised: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.notes or self.accords or self.like)


def tokenize(text: str) -> List[str]:
    folded = ascii_fold(text).lower().replace("'", "")
    folded = _LEADING_MINUS.sub(r"\1 - ", folded)
    tokens: List[str] = []
    for chunk in folded.split():
        if chunk == "-":
            tokens.append("-")
            continue
        tokens.extend(_TOKEN.findall(chunk.replace("-", " ")))
    return tokens


class QueryParser:
    """Dictionary-driven parser.

    ``known_notes`` are canonical note strings.  ``resolve_perfume`` maps a
    list of words to a catalog item id, or None when no perfume carries all
    of those words in its brand and name.
    """

    def __init__(
        self,
        known_notes: Sequence[str],
        resolve_perfume: Optional[Callable[[List[str]], Optional[int]]] = None,
    ) -> None:
        self.phrases: Dict[Tuple[str, ...], Tuple[str, str]] = {}
        for note in known_notes:
            self.phrases[tuple(note.split())] = ("note", note)
        known = set(known_notes)
        for alias, canonical in NOTE_ALIASES.items():
            if canonical in known:
                self.phrases.setdefault(tuple(alias.split()), ("note", canonical))
        # Accord adjectives win over the generic "... notes" note aliases.
        for word, accord in ACCORD_WORDS.items():
            self.phrases[(word,)] = ("accord", accord)
        self.resolve_perfume = resolve_perfume

    def _match_phrase(self, tokens: List[str], start: int) -> Optional[Tuple[int, str, str]]:
        """Longest dictionary phrase starting at ``start``: (length, kind, value)."""
        for length in range(min(MAX_PHRASE_WORDS, len(tokens) - start), 0, -1):
            words = tuple(tokens[start : start + length])
            found = self.phrases.get(words)
            if found is None and length == 1:
                found = self._singular(words[0])
            if found is not None:
                return length, found[0], found[1]
        return None

    def _singular(self, word: str) -> Optional[Tuple[str, str]]:
        for suffix, replacement in [("ies", "y"), ("es", ""), ("s", "")]:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                found = self.phrases.get((word[: -len(suffix)] + replacement,))
                if found is not None:
                    return found
        return None

    def _reference_span(self, tokens: List[str], start: int) -> List[str]:
        words: List[str] = []
        for token in tokens[start : start + MAX_REFERENCE_WORDS]:
            if token in SEPARATORS or token in SCOPE_RESETS or token in NEGATORS:
                break
            words.append(token)
        return words

    def _all_dictionary_words(self, words: List[str]) -> bool:
        position = 0
        while position < len(words):
            if words[position] in STOP_WORDS:
                position += 1
                continue
            match = self._match_phrase(words, position)
            if match is None:
                return False
            position += match[0]
        return True

    def _resolve_reference(
        self, tokens: List[str], start: int, name_only: bool
    ) -> Optional[Tuple[int, int]]:
        """Try to read a perfume name at ``start``: (words consumed, item id)."""
        if self.resolve_perfume is None:
            return None
        span = self._reference_span(tokens, start)
        for length in range(len(span), 0, -1):
            words = span[:length]
            if not name_only and self._all_dictionary_words(words):
                # "I like rose and vanilla" is about notes, not a perfume.
                continue
            item_id = self.resolve_perfume(words)
            if item_id is not None:
                return length, item_id
        return None

    def parse(self, text: str) -> ParsedQuery:
        tokens = tokenize(text)
        parsed = ParsedQuery()
        negated = False
        position = 0

        def add(target: List, value) -> None:
            if value not in target:
                target.append(value)

        while position < len(tokens):
            token = tokens[position]
            following = tokens[position + 1] if position + 1 < len(tokens) else None

            if token in SEPARATORS:
                negated = False
                position += 1
                continue

            # Year filters.
            if token in ("after", "since", "post") and following and _YEAR.match(following):
                parsed.year_min = int(following) + (1 if token == "after" else 0)
                position += 2
                continue
            if token in ("before", "pre") and following and _YEAR.match(following):
                parsed.year_max = int(following) - 1
                position += 2
                continue
            if _DECADE.match(token):
                parsed.year_min = int(token[:4])
                parsed.year_max = int(token[:4]) + 9
                position += 1
                continue

            # Gender filters.
            if token == "unisex":
                add(parsed.genders, "Unisex")
                position += 1
                continue
            if token in MALE_WORDS or token in FEMALE_WORDS:
                add(parsed.genders, "Male" if token in MALE_WORDS else "Female")
                add(parsed.genders, "Unisex")
                position += 1
                continue

            # Reference perfumes: "like X", "similar to X", "dupe of X".
            trigger = next(
                (
                    words
                    for words in REFERENCE_TRIGGERS
                    if tuple(tokens[position : position + len(words)]) == words
                ),
                None,
            )
            if trigger is not None:
                resolved = self._resolve_reference(
                    tokens, position + len(trigger), REFERENCE_TRIGGERS[trigger]
                )
                if resolved is not None:
                    add(parsed.unlike if negated else parsed.like, resolved[1])
                    position += len(trigger) + resolved[0]
                    negated = False
                    continue
                position += len(trigger)
                continue

            if token in NEGATORS:
                negated = True
                position += 1
                continue
            if token in SCOPE_RESETS:
                negated = False
                position += 1
                continue

            match = self._match_phrase(tokens, position)
            if match is not None:
                length, kind, value = match
                if kind == "note":
                    add(parsed.avoid_notes if negated else parsed.notes, value)
                else:
                    add(parsed.avoid_accords if negated else parsed.accords, value)
                position += length
                continue

            if token not in STOP_WORDS:
                add(parsed.unrecognised, token)
            position += 1

        # A term cannot be both wanted and avoided; the exclusion wins.
        parsed.notes = [note for note in parsed.notes if note not in parsed.avoid_notes]
        parsed.accords = [accord for accord in parsed.accords if accord not in parsed.avoid_accords]
        return parsed


def phrase_in_note(phrase_words: Sequence[str], note_words: Sequence[str]) -> bool:
    """True when the phrase occurs in the note as a run of whole words."""
    span = len(phrase_words)
    return any(
        tuple(note_words[start : start + span]) == tuple(phrase_words)
        for start in range(len(note_words) - span + 1)
    )


def notes_matching(phrase: str, vocabulary: Sequence[str]) -> Set[str]:
    """Every note that contains ``phrase`` as whole words ("rose" -> "rose petals")."""
    phrase_words = phrase.split()
    return {note for note in vocabulary if phrase_in_note(phrase_words, note.split())}
