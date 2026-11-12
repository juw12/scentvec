"""Search engine over the perfume catalog.

    engine = ScentEngine.load()
    aventus = engine.lookup("creed aventus")[0]["item_id"]
    engine.similar(aventus, k=10)
    engine.search(text="smoky vanilla and tobacco, not too sweet, no patchouli")

Ranking is cosine similarity in the learned embedding space, plus two small
transparent terms when the request names notes or accords:

    score = similarity + NOTE_MATCH_WEIGHT * (share of requested notes present)
                       + ACCORD_WEIGHT * (wanted accord probability - unwanted)

Every result reports the three parts separately.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .catalog import INDEX_DIR, MODEL_DIR
from .encoder import NoteEncoder, canonical_pyramid
from .query import ParsedQuery, QueryParser, notes_matching
from .textnorm import canonical_notes, make_key

NOTE_MATCH_WEIGHT = 0.30
ACCORD_WEIGHT = 0.50
UNLIKE_WEIGHT = 0.50
MAX_RESULTS = 100
MIN_NOTE_ITEMS_FOR_PARSER = 3
REFERENCE_MIN_RATING_COUNT = 20
# Formats that are rarely what someone means when they type a perfume's name.
SECONDARY_CONCENTRATIONS = frozenset(["aftershave", "mist", "oil", "edc"])

LIST_COLUMNS = [
    "notes_top",
    "notes_middle",
    "notes_base",
    "notes_other",
    "notes_all",
    "accords_voted",
    "accords_predicted",
    "accords_predicted_scores",
    "perfumers",
    "sources",
    "record_ids",
]

# Shorthand people actually type, expanded before name lookup.
NAME_SHORTHAND = {
    "br540": "baccarat rouge 540",
    "br": "baccarat rouge",
    "adg": "acqua di gio",
    "cdnim": "club de nuit intense man",
    "ysl": "yves saint laurent",
    "mfk": "maison francis kurkdjian",
    "jpg": "jean paul gaultier",
    "pdm": "parfums de marly",
    "tf": "tom ford",
    "ck": "calvin klein",
    "d&g": "dolce and gabbana",
    "dg": "dolce gabbana",
}


class EngineError(ValueError):
    """A request the engine cannot answer; the message is safe to show to users."""


@dataclass
class Filters:
    genders: List[str] = field(default_factory=list)
    include_unknown_gender: bool = False
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    brands: List[str] = field(default_factory=list)
    exclude_brands: List[str] = field(default_factory=list)
    concentrations: List[str] = field(default_factory=list)
    families: List[str] = field(default_factory=list)
    min_rating: Optional[float] = None
    min_rating_count: int = 0
    must_notes: List[str] = field(default_factory=list)
    exclude_notes: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Optional[dict]) -> "Filters":
        if not payload:
            return cls()
        if not isinstance(payload, dict):
            raise EngineError("filters must be an object")
        payload = dict(payload)
        for alias, name in FILTER_ALIASES.items():
            if alias in payload:
                if name in payload:
                    raise EngineError(f"give either '{alias}' or '{name}', not both")
                payload[name] = payload.pop(alias)
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise EngineError(f"unknown filter(s): {', '.join(sorted(unknown))}")
        cleaned = {}
        for name, value in payload.items():
            if value is None:
                continue
            if name in _LIST_FILTERS:
                cleaned[name] = _string_list(name, value)
            elif name in _INTEGER_FILTERS:
                cleaned[name] = _number(name, value, whole=True)
            elif name == "min_rating":
                cleaned[name] = _number(name, value, whole=False)
            elif name == "include_unknown_gender":
                if not isinstance(value, bool):
                    raise EngineError("include_unknown_gender must be true or false")
                cleaned[name] = value
        return cls(**cleaned)


FILTER_ALIASES = {
    "gender": "genders",
    "brand": "brands",
    "exclude_brand": "exclude_brands",
    "concentration": "concentrations",
    "family": "families",
    "min_ratings": "min_rating_count",
    "must": "must_notes",
    "avoid": "exclude_notes",
}
_LIST_FILTERS = frozenset(
    ["genders", "brands", "exclude_brands", "concentrations", "families", "must_notes", "exclude_notes"]
)
_INTEGER_FILTERS = frozenset(["year_min", "year_max", "min_rating_count"])
MAX_FILTER_VALUES = 50


def _string_list(name: str, value) -> List[str]:
    """A filter given as one string or a list of strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise EngineError(f"{name} must be a list of strings")
    if len(value) > MAX_FILTER_VALUES:
        raise EngineError(f"{name} accepts at most {MAX_FILTER_VALUES} values")
    result = []
    for entry in value:
        if not isinstance(entry, str):
            raise EngineError(f"{name} must be a list of strings")
        if entry.strip():
            result.append(entry.strip())
    return result


def _number(name: str, value, whole: bool):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EngineError(f"{name} must be a number")
    if not math.isfinite(value):
        raise EngineError(f"{name} must be a finite number")
    if whole:
        if int(value) != value:
            raise EngineError(f"{name} must be a whole number")
        return int(value)
    return float(value)


def _plain(value):
    """Convert numpy / pandas scalars and containers to JSON-safe Python."""
    if isinstance(value, np.ndarray):
        return [_plain(entry) for entry in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if value is pd.NA or value is None:
        return None
    return value


class ScentEngine:
    def __init__(self, encoder: NoteEncoder, catalog: pd.DataFrame, embeddings: np.ndarray) -> None:
        if len(catalog) != len(embeddings):
            raise EngineError("catalog and embedding index have different lengths")
        self.encoder = encoder
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.items: List[dict] = []
        for row in catalog.to_dict("records"):
            item = {name: _plain(value) for name, value in row.items()}
            for column in LIST_COLUMNS:
                item[column] = item.get(column) or []
            if item.get("year") is not None:
                item["year"] = int(item["year"])
            self.items.append(item)
        count = len(self.items)

        vocabulary = encoder.vocabulary
        self.accords = list(vocabulary.accords)
        self.families = list(vocabulary.families)
        self.subfamilies = list(vocabulary.subfamilies)
        scores = encoder.head_scores(self.embeddings)
        self.accord_scores = scores["accords"].astype(np.float32)

        # Columns used by filters.
        self._year = np.array(
            [item["year"] if item["year"] is not None else -1 for item in self.items], dtype=np.int32
        )
        self._rating = np.array(
            [item["rating"] if item["rating"] is not None else np.nan for item in self.items],
            dtype=np.float32,
        )
        self._rating_count = np.array([item["rating_count"] or 0 for item in self.items], dtype=np.int64)
        self._gender = np.array([item["gender"] or "" for item in self.items], dtype=object)
        self._brand_key = np.array([item["brand_key"] for item in self.items], dtype=object)
        self._concentration = np.array(
            [item["concentration"] or "" for item in self.items], dtype=object
        )
        self._family = np.array(
            [item["family"] or item["family_predicted"] for item in self.items], dtype=object
        )
        line_keys = [item["brand_key"] + "|" + item["name_key"] for item in self.items]
        self._line = pd.factorize(pd.Series(line_keys))[0].astype(np.int64)
        self._name_tokens = [tuple(item["name_key"].split()) for item in self.items]
        brand_items: Dict[str, List[int]] = defaultdict(list)
        for item in self.items:
            brand_items[item["brand_key"]].append(item["item_id"])
        self._brand_items = {
            brand: np.asarray(item_ids, dtype=np.int64) for brand, item_ids in brand_items.items()
        }

        # Note -> items, for exclusions, requirements and the lexical bonus.
        note_items: Dict[str, List[int]] = defaultdict(list)
        for item in self.items:
            for note in item["notes_all"]:
                note_items[note].append(item["item_id"])
        self._note_items = {
            note: np.asarray(item_ids, dtype=np.int64) for note, item_ids in note_items.items()
        }
        self._note_vocabulary = sorted(self._note_items)
        self._expansion_cache: Dict[str, Set[str]] = {}

        # Name lookup: word -> items carrying that word in brand or name.
        word_items: Dict[str, Set[int]] = defaultdict(set)
        self._name_words: List[Set[str]] = []
        self._title_words: List[Set[str]] = []
        for item in self.items:
            words = set((item["brand_key"] + " " + item["name_key"]).split())
            self._name_words.append(words)
            self._title_words.append(set(item["name_key"].split()))
            for word in words:
                word_items[word].add(item["item_id"])
        self._word_items = dict(word_items)
        self._words_sorted = sorted(self._word_items)
        self._popularity = np.log1p(self._rating_count.astype(np.float64))
        self._count = count

        parser_notes = [
            note
            for note, item_ids in self._note_items.items()
            if len(item_ids) >= MIN_NOTE_ITEMS_FOR_PARSER
        ]
        self.parser = QueryParser(parser_notes, self._resolve_reference)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR, index_dir: Path = INDEX_DIR) -> "ScentEngine":
        model_dir = Path(model_dir)
        index_dir = Path(index_dir)
        for path in [model_dir / "model.pt", index_dir / "catalog.parquet", index_dir / "embeddings.npy"]:
            if not path.exists():
                raise EngineError(
                    f"missing {path}; run `scentvec train`, `scentvec package` and "
                    "`scentvec build-index` first"
                )
        encoder = NoteEncoder.load(model_dir)
        catalog = pd.read_parquet(index_dir / "catalog.parquet")
        embeddings = np.load(index_dir / "embeddings.npy")
        if embeddings.shape[1] != encoder.dim:
            raise EngineError("the index was built with a different model; rebuild the index")
        return cls(encoder, catalog, embeddings)

    def __len__(self) -> int:
        return self._count

    # --------------------------------------------------------------- name lookup

    def _words_with_prefix(self, prefix: str) -> List[str]:
        start = bisect.bisect_left(self._words_sorted, prefix)
        words = []
        for word in self._words_sorted[start:]:
            if not word.startswith(prefix):
                break
            words.append(word)
        return words

    def _items_for_word(self, word: str, allow_prefix: bool) -> Set[int]:
        exact = self._word_items.get(word)
        if exact is not None and not allow_prefix:
            return exact
        found: Set[int] = set(exact or ())
        if allow_prefix or exact is None:
            for candidate in self._words_with_prefix(word):
                found |= self._word_items[candidate]
        return found

    def _name_query_words(self, query: str) -> List[str]:
        words: List[str] = []
        for word in make_key(query).split():
            words.extend(NAME_SHORTHAND.get(word, word).split())
        return words

    def _tightness_key(self, item_id: int, query_set: Set[str]) -> tuple:
        """Sort key for name matches: the plain perfume before its flankers.

        "creed aventus" should find Aventus itself, not the more-rated
        "Aventus for Her", so unmatched words in the perfume's own name count
        first and popularity only breaks ties.
        """
        unmatched_title_words = len(self._title_words[item_id] - query_set)
        secondary_format = self._concentration[item_id] in SECONDARY_CONCENTRATIONS
        unmatched_words = len(self._name_words[item_id] - query_set)
        return (
            unmatched_title_words,
            secondary_format,
            -self._popularity[item_id],
            unmatched_words,
            item_id,
        )

    def lookup(self, query: str, limit: int = 10) -> List[dict]:
        """Find perfumes by brand and name words; the last word may be a prefix."""
        words = self._name_query_words(query)
        if not words:
            return []
        matches: Dict[int, int] = defaultdict(int)
        for position, word in enumerate(words):
            is_last = position == len(words) - 1
            for item_id in self._items_for_word(word, allow_prefix=is_last):
                matches[item_id] += 1
        if not matches:
            return []
        best = max(matches.values())
        # Keep perfumes matching every word; fall back to the best partial matches.
        candidates = [item_id for item_id, hits in matches.items() if hits == best]
        query_set = set(words)

        def rank_key(item_id: int):
            exact_hits = len(query_set & self._name_words[item_id])
            return (-exact_hits,) + self._tightness_key(item_id, query_set)

        candidates.sort(key=rank_key)
        results = []
        for item_id in candidates[: max(1, min(limit, MAX_RESULTS))]:
            summary = self.summary(item_id)
            summary["matched_all_words"] = best == len(words)
            results.append(summary)
        return results

    def _resolve_reference(self, words: List[str]) -> Optional[int]:
        """Used by the text parser: a perfume whose brand+name has all the words."""
        expanded: List[str] = []
        for word in words:
            expanded.extend(NAME_SHORTHAND.get(word, word).split())
        candidates: Optional[Set[int]] = None
        for word in expanded:
            found = self._word_items.get(word)
            if not found:
                return None
            candidates = set(found) if candidates is None else candidates & found
            if not candidates:
                return None
        if not candidates:
            return None
        query_set = set(expanded)
        best = min(candidates, key=lambda item_id: self._tightness_key(item_id, query_set))
        # A single common word ("rose", "black") must not grab an obscure perfume.
        if len(expanded) == 1 and self._rating_count[best] < REFERENCE_MIN_RATING_COUNT:
            return None
        return int(best)

    # ------------------------------------------------------------------- display

    def _check_item(self, item_id) -> int:
        if isinstance(item_id, bool) or not isinstance(item_id, (int, np.integer)):
            raise EngineError("item id must be an integer")
        if not 0 <= int(item_id) < self._count:
            raise EngineError(f"no perfume with id {item_id}")
        return int(item_id)

    def get(self, item_id: int) -> dict:
        """Full catalog entry."""
        item = dict(self.items[self._check_item(item_id)])
        item["accords"] = self._accords_of(item)
        return item

    def _accords_of(self, item: dict) -> dict:
        if item["accords_voted"]:
            return {"values": item["accords_voted"], "source": "voted"}
        return {"values": item["accords_predicted"], "source": "predicted"}

    def summary(self, item_id: int) -> dict:
        item = self.items[item_id]
        return {
            "item_id": item["item_id"],
            "brand": item["brand"],
            "name": item["name"],
            "year": item["year"],
            "concentration": item["concentration"],
            "gender": item["gender"],
            "notes_top": item["notes_top"],
            "notes_middle": item["notes_middle"],
            "notes_base": item["notes_base"],
            "notes_other": item["notes_other"],
            "accords": self._accords_of(item),
            "family": item["family"] or item["family_predicted"],
            "family_source": "labelled" if item["family"] else "predicted",
            "rating": item["rating"],
            "rating_count": item["rating_count"],
            "url": item["url"],
        }

    # ------------------------------------------------------------------- filters

    def _expand_note(self, phrase: str) -> Set[str]:
        """Catalog notes that contain the phrase as whole words."""
        cached = self._expansion_cache.get(phrase)
        if cached is None:
            cached = notes_matching(phrase, self._note_vocabulary)
            if len(self._expansion_cache) < 5000:
                self._expansion_cache[phrase] = cached
        return cached

    def _items_with_note(self, phrase: str) -> np.ndarray:
        mask = np.zeros(self._count, dtype=bool)
        for note in self._expand_note(phrase):
            mask[self._note_items[note]] = True
        return mask

    def _canonical_list(self, values: Optional[Iterable[str]], what: str) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [part for part in values.split(",")]
        if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
            raise EngineError(f"{what} must be a list of strings")
        return canonical_notes(values)

    def _accord_list(self, values: Optional[Iterable[str]], what: str) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = values.split(",")
        if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
            raise EngineError(f"{what} must be a list of strings")
        by_key = {make_key(accord): accord for accord in self.accords}
        resolved = []
        for value in values:
            accord = by_key.get(make_key(value))
            if accord is None:
                raise EngineError(
                    f"unknown accord '{value}' in {what}; choose from: {', '.join(self.accords)}"
                )
            if accord not in resolved:
                resolved.append(accord)
        return resolved

    def filter_mask(self, filters: Filters) -> np.ndarray:
        mask = np.ones(self._count, dtype=bool)
        if filters.genders:
            wanted = {str(gender).strip().capitalize() for gender in filters.genders}
            unknown = wanted - {"Male", "Female", "Unisex"}
            if unknown:
                raise EngineError("genders must be Male, Female or Unisex")
            gender_mask = np.isin(self._gender, list(wanted))
            if filters.include_unknown_gender:
                gender_mask |= self._gender == ""
            mask &= gender_mask
        if filters.year_min is not None:
            mask &= self._year >= int(filters.year_min)
        if filters.year_max is not None:
            mask &= (self._year <= int(filters.year_max)) & (self._year > 0)
        if filters.brands:
            mask &= np.isin(self._brand_key, [make_key(brand) for brand in filters.brands])
        if filters.exclude_brands:
            mask &= ~np.isin(self._brand_key, [make_key(brand) for brand in filters.exclude_brands])
        if filters.concentrations:
            mask &= np.isin(self._concentration, [str(c).lower() for c in filters.concentrations])
        if filters.families:
            by_key = {make_key(family): family for family in self.families}
            wanted_families = []
            for family in filters.families:
                if make_key(family) not in by_key:
                    raise EngineError(
                        f"unknown family '{family}'; choose from: {', '.join(self.families)}"
                    )
                wanted_families.append(by_key[make_key(family)])
            mask &= np.isin(self._family, wanted_families)
        if filters.min_rating is not None:
            mask &= np.nan_to_num(self._rating, nan=-1.0) >= float(filters.min_rating)
        if filters.min_rating_count:
            mask &= self._rating_count >= int(filters.min_rating_count)
        for note in self._canonical_list(filters.must_notes, "must_notes"):
            mask &= self._items_with_note(note)
        for note in self._canonical_list(filters.exclude_notes, "exclude_notes"):
            mask &= ~self._items_with_note(note)
        return mask

    # ------------------------------------------------------------------- ranking

    def same_line_mask(self, item_id: int) -> np.ndarray:
        """Other versions of one perfume: the same brand, and one name starts with the other.

        "Aventus" therefore covers "Aventus Cologne" and "Aventus 10th
        Anniversary Edition", which are what people mean by "not the same thing
        again" when they ask for alternatives.
        """
        mask = self._line == self._line[item_id]
        tokens = self._name_tokens[item_id]
        if not tokens:
            return mask
        for other in self._brand_items[self._brand_key[item_id]]:
            other_tokens = self._name_tokens[other]
            shorter = min(len(tokens), len(other_tokens))
            if shorter and tokens[:shorter] == other_tokens[:shorter]:
                mask[other] = True
        return mask

    def _top(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        k: int,
        collapse_lines: bool,
        max_per_brand: int,
    ) -> List[int]:
        k = max(1, min(int(k), MAX_RESULTS))
        allowed = np.flatnonzero(mask)
        if len(allowed) == 0:
            return []
        # Take a generous shortlist, then apply the per-line and per-brand caps.
        shortlist_size = min(len(allowed), max(50, k * 20))
        allowed_scores = scores[allowed]
        if shortlist_size < len(allowed):
            part = np.argpartition(-allowed_scores, shortlist_size - 1)[:shortlist_size]
        else:
            part = np.arange(len(allowed))
        order = part[np.lexsort((allowed[part], -allowed_scores[part]))]
        chosen: List[int] = []
        seen_lines: Set[int] = set()
        per_brand: Dict[str, int] = defaultdict(int)
        for position in order:
            item_id = int(allowed[position])
            if collapse_lines:
                if int(self._line[item_id]) in seen_lines:
                    continue
                seen_lines.add(int(self._line[item_id]))
            if max_per_brand > 0:
                brand = self._brand_key[item_id]
                if per_brand[brand] >= max_per_brand:
                    continue
                per_brand[brand] += 1
            chosen.append(item_id)
            if len(chosen) == k:
                break
        return chosen

    def explain(self, first_id: int, second_id: int) -> dict:
        """What two perfumes have in common, in plain terms."""
        first = self.items[self._check_item(first_id)]
        second = self.items[self._check_item(second_id)]
        second_notes = set(second["notes_all"])
        first_accords = self._accords_of(first)["values"]
        second_accords = set(self._accords_of(second)["values"])
        return {
            "similarity": round(float(self.embeddings[first_id] @ self.embeddings[second_id]), 4),
            "shared_notes": [note for note in first["notes_all"] if note in second_notes],
            "only_first": [note for note in first["notes_all"] if note not in second_notes],
            "only_second": [
                note for note in second["notes_all"] if note not in set(first["notes_all"])
            ],
            "shared_accords": [accord for accord in first_accords if accord in second_accords],
        }

    def similar(
        self,
        item_id: int,
        k: int = 10,
        filters: Optional[Filters] = None,
        exclude_same_line: bool = True,
        max_per_brand: int = 0,
    ) -> dict:
        """Nearest perfumes to one catalog entry."""
        item_id = self._check_item(item_id)
        scores = self.embeddings @ self.embeddings[item_id]
        mask = self.filter_mask(filters or Filters())
        mask[item_id] = False
        if exclude_same_line:
            mask &= ~self.same_line_mask(item_id)
        chosen = self._top(scores, mask, k, collapse_lines=True, max_per_brand=max_per_brand)
        results = []
        for other in chosen:
            entry = self.summary(other)
            entry["score"] = round(float(scores[other]), 4)
            explanation = self.explain(item_id, other)
            entry["shared_notes"] = explanation["shared_notes"]
            entry["shared_accords"] = explanation["shared_accords"]
            results.append(entry)
        return {"query": self.summary(item_id), "results": results}

    def describe(
        self,
        top: Optional[Iterable[str]] = None,
        middle: Optional[Iterable[str]] = None,
        base: Optional[Iterable[str]] = None,
        notes: Optional[Iterable[str]] = None,
        include_vector: bool = True,
    ) -> dict:
        """Embed a note list and report how the model reads it."""
        canonical, positions = canonical_pyramid(top, middle, base, notes)
        if not canonical:
            raise EngineError("give at least one note")
        vector = self.encoder.encode([(canonical, positions)])[0]
        scores = self.encoder.head_scores(vector)
        accord_order = np.argsort(-scores["accords"][0])[:6]
        family_order = np.argsort(-scores["family"][0])[:3]
        subfamily_order = np.argsort(-scores["subfamily"][0])[:3]
        result = {
            "notes": [
                {
                    "note": note,
                    "position": ["top", "middle", "base", "unspecified"][position],
                    "recognised_as": self.encoder.known(note),
                }
                for note, position in zip(canonical, positions)
            ],
            "accords": [
                {"accord": self.accords[i], "probability": round(float(scores["accords"][0, i]), 3)}
                for i in accord_order
            ],
            "family": [
                {"family": self.families[i], "probability": round(float(scores["family"][0, i]), 3)}
                for i in family_order
            ],
            "subfamily": [
                {
                    "subfamily": self.subfamilies[i],
                    "probability": round(float(scores["subfamily"][0, i]), 3),
                }
                for i in subfamily_order
            ],
            "dim": int(vector.shape[0]),
        }
        if include_vector:
            result["vector"] = [round(float(value), 6) for value in vector]
        return result

    def search(
        self,
        text: Optional[str] = None,
        top: Optional[Iterable[str]] = None,
        middle: Optional[Iterable[str]] = None,
        base: Optional[Iterable[str]] = None,
        notes: Optional[Iterable[str]] = None,
        accords: Optional[Iterable[str]] = None,
        avoid_notes: Optional[Iterable[str]] = None,
        avoid_accords: Optional[Iterable[str]] = None,
        like: Optional[Sequence[int]] = None,
        unlike: Optional[Sequence[int]] = None,
        like_weights: Optional[Sequence[float]] = None,
        filters: Optional[Filters] = None,
        k: int = 10,
        max_per_brand: int = 0,
    ) -> dict:
        """Search by description: free text, notes, accords and reference perfumes.

        ``like`` with several ids (optionally ``like_weights``) blends perfumes.
        """
        filters = filters or Filters()
        parsed = self.parser.parse(text) if text else ParsedQuery()

        top_notes = self._canonical_list(top, "top")
        middle_notes = self._canonical_list(middle, "middle")
        base_notes = self._canonical_list(base, "base")
        free_notes = self._canonical_list(notes, "notes") + parsed.notes
        wanted_accords = self._accord_list(accords, "accords")
        for accord in parsed.accords:
            if accord not in wanted_accords:
                wanted_accords.append(accord)
        unwanted_accords = self._accord_list(avoid_accords, "avoid_accords")
        for accord in parsed.avoid_accords:
            if accord not in unwanted_accords:
                unwanted_accords.append(accord)
        wanted_accords = [accord for accord in wanted_accords if accord not in unwanted_accords]
        excluded_notes = self._canonical_list(avoid_notes, "avoid_notes")
        for note in parsed.avoid_notes:
            if note not in excluded_notes:
                excluded_notes.append(note)
        like_ids = [self._check_item(item_id) for item_id in (like or [])]
        for item_id in parsed.like:
            if item_id not in like_ids:
                like_ids.append(item_id)
        unlike_ids = [self._check_item(item_id) for item_id in (unlike or [])] + parsed.unlike

        if parsed.genders and not filters.genders:
            filters.genders = list(parsed.genders)
        if parsed.year_min is not None and filters.year_min is None:
            filters.year_min = parsed.year_min
        if parsed.year_max is not None and filters.year_max is None:
            filters.year_max = parsed.year_max

        query_notes, query_positions = canonical_pyramid(top_notes, middle_notes, base_notes, free_notes)
        query_notes_kept = [note for note in query_notes if note not in excluded_notes]
        query_positions = [
            position for note, position in zip(query_notes, query_positions) if note not in excluded_notes
        ]
        query_notes = query_notes_kept
        if not (query_notes or wanted_accords or like_ids):
            hint = ""
            if parsed.unrecognised:
                hint = f" (not understood: {', '.join(parsed.unrecognised)})"
            raise EngineError(
                "nothing to search for: name at least one note, accord or reference perfume" + hint
            )

        # Vector part: requested notes and reference perfumes share one space.
        vector = np.zeros(self.encoder.dim, dtype=np.float32)
        if query_notes:
            vector += self.encoder.encode([(query_notes, query_positions)])[0]
        if like_ids:
            if like_weights is not None:
                if not isinstance(like_weights, (list, tuple)) or not all(
                    isinstance(weight, (int, float)) and not isinstance(weight, bool)
                    for weight in like_weights
                ):
                    raise EngineError("like_weights must be a list of numbers")
                if len(like_weights) != len(like_ids):
                    raise EngineError("like_weights must have one weight per reference perfume")
                weights = np.asarray(like_weights, dtype=np.float32)
                if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                    raise EngineError("like_weights must be non-negative and not all zero")
            else:
                weights = np.ones(len(like_ids), dtype=np.float32)
            blend = (self.embeddings[like_ids] * (weights / weights.sum())[:, None]).sum(axis=0)
            vector += blend / max(float(np.linalg.norm(blend)), 1e-9)
        if unlike_ids:
            vector -= UNLIKE_WEIGHT * self.embeddings[unlike_ids].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        has_vector = norm > 1e-9
        similarity = self.embeddings @ (vector / norm) if has_vector else np.zeros(self._count, np.float32)

        # Lexical part: share of the requested notes that are literally listed.
        note_match = np.zeros(self._count, dtype=np.float32)
        if query_notes:
            for note in query_notes:
                note_match += self._items_with_note(note)
            note_match /= len(query_notes)

        # Accord part.
        accord_part = np.zeros(self._count, dtype=np.float32)
        if wanted_accords:
            columns = [self.accords.index(accord) for accord in wanted_accords]
            accord_part += self.accord_scores[:, columns].mean(axis=1)
        if unwanted_accords:
            columns = [self.accords.index(accord) for accord in unwanted_accords]
            accord_part -= self.accord_scores[:, columns].mean(axis=1)

        accord_weight = ACCORD_WEIGHT if has_vector else 1.0
        scores = similarity + NOTE_MATCH_WEIGHT * note_match + accord_weight * accord_part

        mask = self.filter_mask(filters)
        for note in excluded_notes:
            mask &= ~self._items_with_note(note)
        for item_id in like_ids + unlike_ids:
            mask[item_id] = False
            mask &= ~self.same_line_mask(item_id)

        chosen = self._top(scores, mask, k, collapse_lines=True, max_per_brand=max_per_brand)
        results = []
        for item_id in chosen:
            entry = self.summary(item_id)
            entry["score"] = round(float(scores[item_id]), 4)
            entry["score_parts"] = {
                "similarity": round(float(similarity[item_id]), 4),
                "note_match": round(float(note_match[item_id]), 4),
                "accords": round(float(accord_part[item_id]), 4),
            }
            listed = set(self.items[item_id]["notes_all"])
            entry["matched_notes"] = [
                note for note in query_notes if listed & self._expand_note(note)
            ]
            results.append(entry)

        return {
            "query": {
                "notes": [
                    {"note": note, "position": ["top", "middle", "base", "unspecified"][position]}
                    for note, position in zip(query_notes, query_positions)
                ],
                "accords": wanted_accords,
                "avoid_notes": excluded_notes,
                "avoid_accords": unwanted_accords,
                "like": [self.summary(item_id) for item_id in like_ids],
                "unlike": [self.summary(item_id) for item_id in unlike_ids],
                "filters": {
                    name: value
                    for name, value in filters.__dict__.items()
                    if value not in (None, [], 0, False)
                },
                "unrecognised": parsed.unrecognised,
            },
            "candidates": int(mask.sum()),
            "results": results,
        }

    # --------------------------------------------------------------------- misc

    def suggest_notes(self, prefix: str, limit: int = 12) -> List[dict]:
        """Note autocomplete, most used first."""
        key = make_key(prefix)
        if not key:
            return []
        found = [
            (len(self._note_items[note]), note)
            for note in self._note_vocabulary
            if note.startswith(key) or (" " + key) in note
        ]
        found.sort(key=lambda entry: (-entry[0], entry[1]))
        return [{"note": note, "perfumes": count} for count, note in found[:limit]]

    def stats(self) -> dict:
        return {
            "perfumes": self._count,
            "brands": int(len(set(self._brand_key))),
            "distinct_notes": len(self._note_vocabulary),
            "embedding_dim": int(self.encoder.dim),
            "accords": self.accords,
            "families": self.families,
        }


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
