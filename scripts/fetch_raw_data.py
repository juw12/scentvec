#!/usr/bin/env python3
"""Download the raw tables scentvec is built from into data/raw.

    python3 scripts/fetch_raw_data.py

The tables are not redistributed with this repository because their reuse terms
differ (see MODEL_CARD.md).  Every URL is pinned to a fixed revision and every
file is checked against the SHA-256 of the copy the published model was trained
on, so a successful run reproduces the training inputs exactly.  Files that are
already present and correct are skipped.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# (file name in data/raw, pinned source URL, expected SHA-256)
SOURCES = [
    (
        "parfumo_data_clean.csv",
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/54ea581e1352f6c55b9343feb77df0fc29c78621/data/2024/2024-12-10/parfumo_data_clean.csv",
        "94956339772a02c0355ff3f9de4f15cd3f078a124b2829dde30bcee83446cc7d",
    ),
    (
        "doevent_perfumes.jsonl",
        "https://huggingface.co/datasets/doevent/perfume/resolve/9d2db452d8f0d649c4bfb0189a2fcc5282028a3c/perfumes.jsonl",
        "355eb32ac4b76bdc8e1351d5a46bbd5e8f712c62f209819191c88cede5942db9",
    ),
    (
        "flavorgraph_nodes_191120.csv",
        "https://raw.githubusercontent.com/lamypark/FlavorGraph/8d3472d0823f4542b2ce47f83e28046c41d2f06a/input/nodes_191120.csv",
        "ea59d1858ae3a41de14aeabda32e6274693ea7897eb6ae1a3b7262bac5526d5a",
    ),
    (
        "flavorgraph_edges_191120.csv",
        "https://raw.githubusercontent.com/lamypark/FlavorGraph/8d3472d0823f4542b2ce47f83e28046c41d2f06a/input/edges_191120.csv",
        "af2c2324f34fc44493ecf2711787a50f16cc8c61c474dfae64194517a3f51399",
    ),
    (
        "goodscents_behavior.csv",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/goodscents/behavior.csv",
        "18968aa66cab974ee2f942ad0abd0875ccf2bcc9501c1cb306a54bd639a22219",
    ),
    (
        "goodscents_molecules.csv",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/goodscents/molecules.csv",
        "0d9bd38c54a40a051ae7b771d51f11ac1fcf021e0369fc53583e5f634e6e8bb4",
    ),
    (
        "goodscents_stimuli.csv",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/goodscents/stimuli.csv",
        "356cdd74a859ab1e8e061952dbb21587afc8981bb460cc670481d688afac923f",
    ),
    (
        "goodscents_manifest.toml",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/goodscents/manifest.toml",
        "eda03e25bbcd763c8c7286b908822776756649a858ee9727c31d69754ca2e46d",
    ),
    (
        "leffingwell_behavior.csv",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/leffingwell/behavior.csv",
        "afe0846b159ac00691ba4c07074829f7ff262b80fa32a3280661457153e70c41",
    ),
    (
        "leffingwell_molecules.csv",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/leffingwell/molecules.csv",
        "a06841a275c35b9db9253a24ac913ad97bf39f2c778e4f8a5c4df641e7dc0fa6",
    ),
    (
        "leffingwell_stimuli.csv",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/leffingwell/stimuli.csv",
        "ff4be9ba6a66523bc5a165ded09eee125ccb79d4988090d682b323d9331a44c8",
    ),
    (
        "leffingwell_manifest.toml",
        "https://raw.githubusercontent.com/pyrfume/pyrfume-data/8054ea98ed675005ec10e67359902f500e4911b0/leffingwell/manifest.toml",
        "341039ecb58ca1043012485b6b4f127425dd5def21caa89c67483dc1e31945ab",
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "scentvec-fetch"})
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
    partial.replace(target)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, url, expected in SOURCES:
        target = RAW_DIR / name
        if target.exists() and sha256_of(target) == expected:
            print(f"ok       {name}")
            continue
        print(f"fetching {name}", flush=True)
        try:
            download(url, target)
        except OSError as error:
            failures.append(f"{name}: download failed ({error})")
            continue
        actual = sha256_of(target)
        if actual != expected:
            failures.append(f"{name}: checksum mismatch (got {actual})")
    if failures:
        print("\nNot all raw files are in place:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\nAll {len(SOURCES)} raw files are in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
