"""Odor-chemistry prior for fragrance notes.

A perfume note such as "sandalwood" is a marketing word, not a molecule.
This module ties notes to measured odor chemistry so that the encoder can
start from something better than a random vector for notes it rarely sees.

Data (pyrfume-data, research use; see MODEL_CARD.md):

* GoodScents + Leffingwell ("GS-LF"): about 6,000 molecules, each tagged by
  perfumers with odor descriptors ("woody", "creamy", "rose", ...).
* FlavorGraph (Apache-2.0): food ingredient -> flavor compound edges.

Every key (a descriptor, a molecule name or a FlavorGraph ingredient) gets a
*descriptor profile*: which odor descriptors co-occur with it, more than
chance would predict.  Three routes produce a profile for a note:

``molecule``     the note names an aroma chemical ("cashmeran", "coumarin")
``descriptor``   the note is itself an odor descriptor ("sandalwood", "musk");
                 its profile is the average of the molecules perfumers
                 describe with that word
``ingredient``   the note is a FlavorGraph ingredient ("ginger"); its profile
                 is the average over its labelled flavor compounds
``token``        fallback: average the profiles of the note's words

The mapping is noisy by construction and is ablated in the evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import PROCESSED_DIR, RAW_DIR
from .textnorm import make_key

MIN_MOLECULES_PER_DESCRIPTOR = 20
MIN_MOLECULES_PER_KEY = 3
CHEM_TABLE_FILE = "chem_table.npz"

# Perfume-note spelling -> GS-LF descriptor spelling.
DESCRIPTOR_ALIASES = {
    "cedarwood": "cedar",
    "incense": "incense",
    "oud": "woody",
    "iris": "orris",
    "lily of the valley": "muguet",
    "blackcurrant": "black currant",
    "blackcurrant bud": "black currant",
    "ambergris": "ambergris",
    "ambroxan": "ambergris",
    "oakmoss": "mossy",
    "moss": "mossy",
    "treemoss": "mossy",
    "leather": "leathery",
    "suede": "leathery",
    "aldehydes": "aldehydic",
    "marine notes": "marine",
    "ozonic notes": "ozone",
    "tonka bean": "tonka",
    "orange blossom": "orangeflower",
    "tobacco leaf": "tobacco",
    "resins": "resinous",
    "smoke": "smoky",
    "herbs": "herbal",
    "balsam": "balsamic",
    "earth": "earthy",
    "powder": "powdery",
    "milk": "milky",
    "cream": "creamy",
    "butter": "buttery",
    "nuts": "nutty",
    "animal notes": "animal",
    "animalic notes": "animal",
    "grass": "grassy",
    "wax": "waxy",
    "metal": "metallic",
    "soap": "soapy",
}


def _descriptor_key(text: str) -> str:
    return make_key(text)


def load_labelled_molecules(raw_dir: Path = RAW_DIR) -> Tuple[Dict[int, set], Dict[int, str]]:
    """Return ({cid: descriptors}, {cid: name}) merged from GoodScents and Leffingwell."""
    labels: Dict[int, set] = defaultdict(set)
    names: Dict[int, str] = {}

    leffingwell = pd.read_csv(raw_dir / "leffingwell_behavior.csv")
    descriptor_columns = [column for column in leffingwell.columns if column != "Stimulus"]
    values = leffingwell[descriptor_columns].to_numpy()
    for cid, row in zip(leffingwell["Stimulus"], values):
        for column_index in np.flatnonzero(row):
            labels[int(cid)].add(_descriptor_key(descriptor_columns[column_index]))
    for cid, name in pd.read_csv(raw_dir / "leffingwell_molecules.csv")[["CID", "name"]].values:
        if isinstance(name, str):
            names[int(cid)] = name

    stimuli = pd.read_csv(raw_dir / "goodscents_stimuli.csv")
    stimulus_to_cid = dict(zip(stimuli["Stimulus"], stimuli["CID"]))
    behavior = pd.read_csv(raw_dir / "goodscents_behavior.csv")
    for stimulus, descriptors in zip(behavior["Stimulus"], behavior["Descriptors"]):
        cid = stimulus_to_cid.get(stimulus)
        if cid is None or pd.isna(cid) or not isinstance(descriptors, str):
            continue
        for descriptor in descriptors.split(";"):
            key = _descriptor_key(descriptor)
            if key:
                labels[int(cid)].add(key)
    for cid, name in pd.read_csv(raw_dir / "goodscents_molecules.csv")[["CID", "name"]].values:
        if isinstance(name, str):
            names.setdefault(int(cid), name)

    labels = {cid: descriptors - {"odorless"} for cid, descriptors in labels.items()}
    labels = {cid: descriptors for cid, descriptors in labels.items() if descriptors}
    return labels, names


def load_flavorgraph(raw_dir: Path = RAW_DIR) -> Dict[str, List[int]]:
    """Return {ingredient name: [PubChem CID, ...]} from FlavorGraph compound edges."""
    nodes = pd.read_csv(raw_dir / "flavorgraph_nodes_191120.csv")
    edges = pd.read_csv(raw_dir / "flavorgraph_edges_191120.csv")
    edges = edges[edges["edge_type"].isin(["ingr-fcomp", "ingr-dcomp"])]
    node_type = dict(zip(nodes["node_id"], nodes["node_type"]))
    node_name = dict(zip(nodes["node_id"], nodes["name"]))
    node_cid = dict(zip(nodes["node_id"], pd.to_numeric(nodes["id"], errors="coerce")))

    compounds: Dict[str, List[int]] = defaultdict(list)
    for first, second in zip(edges["id_1"], edges["id_2"]):
        if node_type.get(first) == "compound":
            first, second = second, first
        if node_type.get(first) != "ingredient" or node_type.get(second) != "compound":
            continue
        cid = node_cid.get(second)
        if cid is None or pd.isna(cid):
            continue
        compounds[make_key(str(node_name[first]).replace("_", " "))].append(int(cid))
    return dict(compounds)


class ChemTable:
    """Descriptor profiles keyed by descriptor, molecule or ingredient name."""

    def __init__(
        self,
        descriptors: List[str],
        keys: List[str],
        routes: List[str],
        matrix: np.ndarray,
    ) -> None:
        self.descriptors = descriptors
        self.keys = keys
        self.routes = routes
        self.matrix = matrix.astype(np.float32)
        self._row: Dict[str, int] = {key: row for row, key in enumerate(keys)}
        self._cache: Dict[str, Tuple[Optional[np.ndarray], str]] = {}

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            descriptors=np.array(self.descriptors),
            keys=np.array(self.keys),
            routes=np.array(self.routes),
            matrix=self.matrix,
        )

    @classmethod
    def load(cls, path: Path) -> "ChemTable":
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                [str(item) for item in archive["descriptors"]],
                [str(item) for item in archive["keys"]],
                [str(item) for item in archive["routes"]],
                archive["matrix"],
            )

    def _exact(self, key: str) -> Optional[int]:
        key = DESCRIPTOR_ALIASES.get(key, key)
        if key in self._row:
            return self._row[key]
        if key.endswith(" notes") and key[:-6] in self._row:
            return self._row[key[:-6]]
        return None

    def lookup(self, note: str) -> Tuple[Optional[np.ndarray], str]:
        """Return (profile or None, route) for a canonical note."""
        if note in self._cache:
            return self._cache[note]
        row = self._exact(note)
        if row is not None:
            result = (self.matrix[row], self.routes[row])
        else:
            token_rows = [self._exact(token) for token in note.split()]
            token_rows = [token_row for token_row in token_rows if token_row is not None]
            if token_rows:
                profile = self.matrix[token_rows].mean(axis=0)
                norm = float(np.linalg.norm(profile))
                result = (profile / norm if norm > 0 else None, "token")
            else:
                result = (None, "none")
        self._cache[note] = result
        return result

    def features(self, notes: Iterable[str]) -> np.ndarray:
        """Stack profiles for a list of notes; unmapped notes are all-zero rows."""
        notes = list(notes)
        output = np.zeros((len(notes), self.dim), dtype=np.float32)
        for position, note in enumerate(notes):
            profile, _ = self.lookup(note)
            if profile is not None:
                output[position] = profile
        return output

    def top_descriptors(self, note: str, count: int = 6) -> List[str]:
        profile, _ = self.lookup(note)
        if profile is None:
            return []
        order = np.argsort(-profile)[:count]
        return [self.descriptors[index] for index in order if profile[index] > 0]


def _ppmi_profile(member_rows: np.ndarray, base_rate: np.ndarray) -> np.ndarray:
    """Positive pointwise mutual information of descriptors within a group of molecules."""
    # Additive smoothing keeps tiny groups from producing extreme lifts.
    rate = (member_rows.sum(axis=0) + 0.5 * base_rate) / (member_rows.shape[0] + 0.5)
    profile = np.maximum(np.log(rate / base_rate), 0.0)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def build_chem_table(raw_dir: Path = RAW_DIR) -> ChemTable:
    labels, names = load_labelled_molecules(raw_dir)
    descriptor_counts = Counter(
        descriptor for descriptors in labels.values() for descriptor in descriptors
    )
    descriptors = sorted(
        descriptor
        for descriptor, count in descriptor_counts.items()
        if count >= MIN_MOLECULES_PER_DESCRIPTOR
    )
    column = {descriptor: index for index, descriptor in enumerate(descriptors)}

    cids = sorted(labels)
    cid_row = {cid: row for row, cid in enumerate(cids)}
    multi_hot = np.zeros((len(cids), len(descriptors)), dtype=np.float32)
    for cid in cids:
        for descriptor in labels[cid]:
            if descriptor in column:
                multi_hot[cid_row[cid], column[descriptor]] = 1.0
    base_rate = (multi_hot.sum(axis=0) + 1.0) / (len(cids) + 1.0)

    keys: List[str] = []
    routes: List[str] = []
    profiles: List[np.ndarray] = []
    seen = set()

    def add(key: str, route: str, member_rows: np.ndarray) -> None:
        if not key or key in seen:
            return
        profile = _ppmi_profile(member_rows, base_rate)
        if not np.any(profile):
            return
        seen.add(key)
        keys.append(key)
        routes.append(route)
        profiles.append(profile)

    # Route priority is the insertion order: molecule, descriptor, ingredient.
    for cid in cids:
        name = names.get(cid)
        if name:
            add(make_key(name), "molecule", multi_hot[[cid_row[cid]]])

    members_by_descriptor: Dict[str, List[int]] = defaultdict(list)
    for cid in cids:
        for descriptor in labels[cid]:
            members_by_descriptor[descriptor].append(cid_row[cid])
    for descriptor, members in sorted(members_by_descriptor.items()):
        if len(members) >= MIN_MOLECULES_PER_KEY:
            add(descriptor, "descriptor", multi_hot[members])

    for ingredient, compound_cids in sorted(load_flavorgraph(raw_dir).items()):
        members = sorted({cid_row[cid] for cid in compound_cids if cid in cid_row})
        if len(members) >= MIN_MOLECULES_PER_KEY:
            add(ingredient, "ingredient", multi_hot[members])

    return ChemTable(descriptors, keys, routes, np.vstack(profiles))


def coverage_report(table: ChemTable, note_counts: Counter) -> dict:
    by_route_notes: Counter = Counter()
    by_route_occurrences: Counter = Counter()
    for note, count in note_counts.items():
        _, route = table.lookup(note)
        by_route_notes[route] += 1
        by_route_occurrences[route] += count
    total_occurrences = sum(note_counts.values())
    return {
        "descriptor_dimensions": table.dim,
        "table_keys": len(table.keys),
        "distinct_notes": len(note_counts),
        "notes_by_route": dict(by_route_notes),
        "share_of_note_occurrences_by_route": {
            route: round(count / total_occurrences, 4)
            for route, count in by_route_occurrences.items()
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the note odor-chemistry table.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    arguments = parser.parse_args(argv)

    table = build_chem_table(arguments.raw_dir)
    table.save(arguments.out_dir / CHEM_TABLE_FILE)

    records = pd.read_parquet(arguments.out_dir / "records.parquet")
    note_counts: Counter = Counter()
    for column_name in ["notes_top", "notes_middle", "notes_base", "notes_flat"]:
        for notes in records[column_name]:
            note_counts.update(notes)
    report = coverage_report(table, note_counts)
    (arguments.out_dir / "chem_coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for note in ["sandalwood", "musk", "bergamot", "cashmeran", "ginger", "white musk", "oud"]:
        print(f"{note:12s} [{table.lookup(note)[1]}] {table.top_descriptors(note, 8)}")
    unmapped = [
        (note, count) for note, count in note_counts.most_common(400)
        if table.lookup(note)[1] == "none"
    ]
    print("most frequent unmapped:", unmapped[:40])


if __name__ == "__main__":
    main()
