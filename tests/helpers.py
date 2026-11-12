"""A tiny, self-contained engine so the tests need no trained model or data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scentvec.chem import ChemTable  # noqa: E402
from scentvec.data import ACCORD_VOCABULARY  # noqa: E402
from scentvec.encoder import NoteEncoder, canonical_pyramid  # noqa: E402
from scentvec.engine import ScentEngine  # noqa: E402
from scentvec.features import Vocabulary  # noqa: E402
from scentvec.model import ModelConfig, ScentEncoder  # noqa: E402
from scentvec.textnorm import make_key  # noqa: E402

NOTES = [
    "amber",
    "bergamot",
    "birch",
    "blackcurrant",
    "cedarwood",
    "incense",
    "jasmine",
    "lemon",
    "musk",
    "oakmoss",
    "patchouli",
    "pineapple",
    "rose",
    "saffron",
    "tobacco",
    "vanilla",
]
FAMILIES = ["Citrus", "Floral", "Oriental", "Woody"]
SUBFAMILIES = ["Citrus Aromatic", "Floral Fruity", "Oriental Vanilla", "Woody Chypre"]

# brand, name, year, concentration, gender, top, middle, base, ratings
PERFUMES = [
    ("Creed", "Aventus", 2010, "edp", "Male", ["pineapple", "bergamot", "blackcurrant"], ["birch", "jasmine"], ["musk", "oakmoss", "vanilla"], 5000),
    ("Creed", "Aventus for Her", 2016, "edp", "Female", ["bergamot", "lemon"], ["rose", "jasmine"], ["musk", "amber"], 9000),
    ("Creed", "Aventus Cologne", 2019, "edp", "Male", ["lemon", "bergamot"], ["patchouli"], ["musk", "birch"], 800),
    ("Armaf", "Club de Nuit Intense Man", 2015, "edt", "Male", ["pineapple", "lemon", "blackcurrant"], ["birch", "jasmine", "rose"], ["musk", "vanilla", "patchouli"], 3000),
    ("Maison Francis Kurkdjian", "Baccarat Rouge 540", 2015, "edp", "Unisex", ["saffron", "jasmine"], ["amber"], ["cedarwood"], 7000),
    ("Tom Ford", "Tobacco Vanille", 2007, "edp", "Unisex", ["tobacco"], ["vanilla"], ["amber"], 6000),
    ("Chanel", "N°5", 1921, "edp", "Female", ["lemon", "bergamot"], ["jasmine", "rose"], ["vanilla", "musk"], 4000),
    ("Nobody", "Quiet Rose", None, None, None, ["rose"], [], ["musk"], 3),
]


def build_tiny_encoder(seed: int = 0) -> NoteEncoder:
    torch.manual_seed(seed)
    tokens = sorted({token for note in NOTES for token in note.split()})
    vocabulary = Vocabulary(list(NOTES), tokens, list(ACCORD_VOCABULARY), FAMILIES, SUBFAMILIES)
    generator = np.random.default_rng(seed)
    chem_matrix = np.abs(generator.normal(size=(len(NOTES), 6))).astype(np.float32)
    chem_matrix /= np.linalg.norm(chem_matrix, axis=1, keepdims=True)
    chem_table = ChemTable(
        descriptors=["citrus", "floral", "woody", "sweet", "smoky", "musky"],
        keys=list(NOTES),
        routes=["descriptor"] * len(NOTES),
        matrix=chem_matrix,
    )
    config = ModelConfig(
        n_note_ids=vocabulary.n_note_ids,
        n_token_ids=vocabulary.n_token_ids,
        chem_dim=chem_table.dim,
        n_accords=len(vocabulary.accords),
        n_families=len(FAMILIES),
        n_subfamilies=len(SUBFAMILIES),
        hidden_dim=32,
        embed_dim=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        note_unk_dropout=0.0,
    )
    return NoteEncoder(ScentEncoder(config), vocabulary, chem_table)


def build_tiny_engine() -> ScentEngine:
    encoder = build_tiny_encoder()
    items: List[dict] = []
    pyramids = []
    for item_id, (brand, name, year, concentration, gender, top, middle, base, ratings) in enumerate(PERFUMES):
        pyramids.append(canonical_pyramid(top, middle, base, []))
        items.append(
            {
                "item_id": item_id,
                "brand": brand,
                "name": name,
                "brand_key": make_key(brand),
                "name_key": make_key(name),
                "year": year,
                "concentration": concentration,
                "gender": gender,
                "notes_top": top,
                "notes_middle": middle,
                "notes_base": base,
                "notes_other": [],
                "notes_all": top + middle + base,
                "accords_voted": ["Fresh"] if item_id % 2 == 0 else [],
                "family": FAMILIES[item_id % len(FAMILIES)],
                "subfamily": SUBFAMILIES[item_id % len(SUBFAMILIES)],
                "rating": 8.0 if ratings > 100 else None,
                "rating_count": ratings,
                "perfumers": [],
                "url": None,
                "description": None,
                "sources": ["parfumo"],
                "record_ids": [f"p{item_id}"],
                "split": "train",
            }
        )
    embeddings = encoder.encode(pyramids)
    scores = encoder.head_scores(embeddings)
    for position, item in enumerate(items):
        order = np.argsort(-scores["accords"][position])[:5]
        item["accords_predicted"] = [ACCORD_VOCABULARY[index] for index in order]
        item["accords_predicted_scores"] = [float(scores["accords"][position, index]) for index in order]
        item["family_predicted"] = FAMILIES[int(scores["family"][position].argmax())]
        item["subfamily_predicted"] = SUBFAMILIES[int(scores["subfamily"][position].argmax())]
    return ScentEngine(encoder, pd.DataFrame(items), embeddings)
