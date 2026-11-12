"""Text normalisation for perfume names, brands and fragrance notes.

Two perfume databases never spell a note the same way: one writes
"Haitian vetiver", the other "Vetiver (Haiti)"; one says "Frankincense", the
other "Incense / Olibanum".  Everything downstream (vocabulary, baselines,
the encoder) works on the canonical form produced here, so the comparison
between the learned model and the lexical baselines stays fair.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

# Origin words describe where a raw material was grown, not what it smells
# like at the granularity this model works at, so they are dropped.
GEOGRAPHIC_WORDS = frozenset(
    """
    african alaskan algerian amalfi american arabian argentinian asian assam atlas australian
    balinese bengal borneo bourbon brazil brazilian british bulgarian burmese calabria calabrian
    caledonian californian cambodian canadian capri caribbean ceylon ceylonese china chinese
    comorian comoros corsican cuban cypriot dutch east eastern ecuadorian egypt egyptian english
    ethiopian european florentine florida floridian french georgian german grasse greek
    guatemalan haiti haitian hawaiian himalayan india indian indonesian iranian isparta italian
    italy jamaican japan japanese java javanese kannauj kenyan laos laotian lebanese madagascan
    madagascar malabar malayan malaysian mauritian mediterranean mexican moroccan morocco mysore
    nepalese nyon aop oman omani paraguayan persian peru peruvian philippine
    provencal provence reunion russian sardinian siam siamese siberian sicilian sicily singapore
    somali somalian spanish sri lankan sumatran taif tahiti tahitian texan texas thai tibetan
    tunisian turkish tuscan virginia virginian west western yemeni zanzibar
    colombian guatemala indonesia lanka nepal paraguay polynesian sumatra tanzanian ugandan
    venezuela venezuelan vietnamese
    """.split()
)

# Words that describe the extraction form rather than the smell.
FORM_WORDS = frozenset(
    """
    absolute absolu accord co2 concrete essence essential extract infusion oil resinoid tincture
    """.split()
)

# Botanical qualifiers worth keeping when they appear in parentheses.
KEPT_QUALIFIERS = frozenset(["sambac", "grandiflorum", "damascena", "centifolia", "tuberosa"])

# Cross-database synonyms, applied after the generic cleaning steps.
NOTE_ALIASES = {
    "mandarin orange": "mandarin",
    "mandarine": "mandarin",
    "cedar": "cedarwood",
    "cedar wood": "cedarwood",
    "frankincense": "incense",
    "olibanum": "incense",
    "incense olibanum": "incense",
    "agarwood": "oud",
    "oud agarwood": "oud",
    "agarwood oud": "oud",
    "oudh": "oud",
    "aoud": "oud",
    "iris orris": "iris",
    "orris": "iris",
    "orris root": "iris",
    "iris root": "iris",
    "ciste labdanum": "labdanum",
    "cistus labdanum": "labdanum",
    "gaiac wood": "guaiac wood",
    "gaiacwood": "guaiac wood",
    "guaiacwood": "guaiac wood",
    "violet leaves": "violet leaf",
    "patchouli leaf": "patchouli",
    "patchouly": "patchouli",
    "tonka": "tonka bean",
    "tonka beans": "tonka bean",
    "ylang": "ylang ylang",
    "citruses": "citrus notes",
    "citrus": "citrus notes",
    "citrus fruits": "citrus notes",
    "woods": "woody notes",
    "wood": "woody notes",
    "woody": "woody notes",
    "spices": "spicy notes",
    "spicy": "spicy notes",
    "flowers": "floral notes",
    "floral": "floral notes",
    "fruits": "fruity notes",
    "fruity": "fruity notes",
    "green": "green notes",
    "musks": "musk",
    "ambroxide": "ambroxan",
    "ambrox": "ambroxan",
    "black currant": "blackcurrant",
    "cassis": "blackcurrant",
    "lily of valley": "lily of the valley",
    "muguet": "lily of the valley",
    "vanille": "vanilla",
    "bergamote": "bergamot",
    "sandal": "sandalwood",
    "vetyver": "vetiver",
    "oak moss": "oakmoss",
    "tree moss": "treemoss",
    "sea notes": "marine notes",
    "marine": "marine notes",
    "aquatic notes": "marine notes",
    "ozone": "ozonic notes",
    "ozonic": "ozonic notes",
    # Added in the refinement pass, from disagreements between the two
    # databases on training-split perfumes they both describe.
    "elemi resin": "elemi",
    "juniper berries": "juniper berry",
    "cypriol nagarmota": "cypriol",
    "cypriol nagarmotha": "cypriol",
    "nagarmotha": "cypriol",
    "damask rose": "rose damascena",
    "rosa centifolia": "rose centifolia",
    "may rose": "rose centifolia",
    "jasminum grandiflorum": "jasmine grandiflorum",
    "palisander rosewood": "rosewood",
    "ambrette seed": "ambrette",
    "ambrette seeds": "ambrette",
    "coriander seed": "coriander",
    "coriander seeds": "coriander",
    "mirabelle plum": "mirabelle",
    "salty caramel": "salted caramel",
    "aldehyde": "aldehydes",
    "petigrain": "petitgrain",
    "cocoa bean": "cocoa",
    "coffee bean": "coffee",
    "coffee beans": "coffee",
    "primofiore lemon": "lemon",
    "anjou pear": "pear",
    "williams pear": "pear",
    "nashi pear": "pear",
    "granny smith apple": "green apple",
    "woodland strawberry": "strawberry",
    "key lime": "lime",
    "everlasting flower": "immortelle",
    "cashmere wood": "cashmeran",
    "musky notes": "musk",
    "gourmand": "gourmand notes",
    "bay": "bay leaf",
    "birch wood": "birch",
}

# Rewrites inside longer notes, applied before the alias lookup:
# "green mandarin orange" -> "green mandarin".
PHRASE_REWRITES = [
    (re.compile(r"\bmandarin orange\b"), "mandarin"),
]

_PARENTHESIS = re.compile(r"\(([^)]*)\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SPACES = re.compile(r"\s+")
_NUMERO = re.compile(r"\bn\s*[°ºo]\s*(?=\d)")

CONCENTRATION_PATTERNS = [
    ("extrait", re.compile(r"\b(extrait de parfum|extrait|pure parfum|parfum extrait)\b")),
    ("edp", re.compile(r"\b(eau de parfum|edp)\b")),
    ("edt", re.compile(r"\b(eau de toilette|edt)\b")),
    ("edc", re.compile(r"\b(eau de cologne|edc)\b")),
    ("oil", re.compile(r"\b(perfume oil|parfum oil|attar)\b")),
    ("mist", re.compile(r"\b(body spray|body mist|hair mist|hair perfume|body and hair spray)\b")),
    ("aftershave", re.compile(r"\b(after shave lotion|after shave|aftershave)\b")),
    ("parfum", re.compile(r"\b(parfum|perfume)\b")),
]


def ascii_fold(text: str) -> str:
    """Strip accents and compatibility characters: 'Fougère' -> 'Fougere'."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def make_key(text: object) -> str:
    """Lowercase alphanumeric key used to match brands and perfume names."""
    folded = ascii_fold(str(text)).lower().replace("&", " and ")
    folded = _NUMERO.sub("no ", folded)
    folded = _NON_ALNUM.sub(" ", folded)
    return _SPACES.sub(" ", folded).strip()


def detect_concentration(text: object) -> Optional[str]:
    """Return a short concentration code ('edp', 'edt', ...) if one is named."""
    if not isinstance(text, str) or not text:
        return None
    key = make_key(text)
    for code, pattern in CONCENTRATION_PATTERNS:
        if pattern.search(key):
            return code
    return None


def strip_concentration(key: str) -> str:
    """Remove concentration wording from an already-normalised name key."""
    for _, pattern in CONCENTRATION_PATTERNS[:-1]:
        key = pattern.sub(" ", key)
    return _SPACES.sub(" ", key).strip()


def canonical_note(raw: object) -> Optional[str]:
    """Map a raw note string to its canonical form, or None if nothing is left.

    >>> canonical_note("Vetiver (Haiti)")
    'vetiver'
    >>> canonical_note("Jasmine (Sambac)")
    'jasmine sambac'
    >>> canonical_note("Incense / Olibanum")
    'incense'
    >>> canonical_note("Clearwood™")
    'clearwood'
    """
    if not isinstance(raw, str):
        return None
    text = ascii_fold(raw.replace("™", " ").replace("®", " ")).lower()

    kept_qualifiers: List[str] = []
    for qualifier in _PARENTHESIS.findall(text):
        for token in _NON_ALNUM.sub(" ", qualifier).split():
            if token in KEPT_QUALIFIERS:
                kept_qualifiers.append(token)
    text = _PARENTHESIS.sub(" ", text)

    tokens = [token for token in _NON_ALNUM.sub(" ", text).split() if token]
    tokens = [token for token in tokens if token not in FORM_WORDS]
    stripped = [token for token in tokens if token not in GEOGRAPHIC_WORDS]
    # A note made only of origin words ("Java") would vanish entirely; keep it.
    tokens = stripped or tokens
    tokens.extend(qualifier for qualifier in kept_qualifiers if qualifier not in tokens)
    if not tokens:
        return None

    note = " ".join(tokens)
    for pattern, replacement in PHRASE_REWRITES:
        note = pattern.sub(replacement, note)
    return NOTE_ALIASES.get(note, note)


def canonical_notes(raw_notes: Iterable[object]) -> List[str]:
    """Canonicalise a list of notes, dropping empties and duplicates in order."""
    seen = set()
    result: List[str] = []
    for raw in raw_notes:
        note = canonical_note(raw)
        if note and note not in seen:
            seen.add(note)
            result.append(note)
    return result


def split_note_cell(cell: object) -> List[str]:
    """Split a comma-separated note cell from a CSV into raw note strings."""
    if not isinstance(cell, str):
        return []
    return [part.strip() for part in cell.split(",") if part.strip()]


def note_tokens(note: str) -> List[str]:
    """Word tokens of a canonical note, used for compositional embeddings."""
    return note.split()
