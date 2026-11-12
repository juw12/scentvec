# scentvec

An embedding model and search engine for **finished perfumes**.

scentvec turns a perfume's note pyramid (top, heart and base notes) into one
128-dimensional unit vector. Perfumes with similar vectors have similar listed
compositions. On top of the model there is a catalog of 44,658 perfumes with a
command line, a local web app, a JSON API and a Python API.

What it is good for:

- "What is similar to X?" over 44,658 perfumes from 1,888 brands.
- "Find me a smoky vanilla tobacco that is not too sweet, no patchouli."
- "Something between perfume A and perfume B."
- Embedding a note list that is not in the catalog (your own formula or brief)
  and predicting its accords and olfactive family.

What it is not: a model of how a perfume actually smells on skin. It only sees
the **published note list**. Two perfumes that smell alike but publish different
notes will be far apart. See [MODEL_CARD.md](MODEL_CARD.md) for measured
performance and limits.

## Quick start

Requires Python 3.9 or newer, git, and about 1.5 GB of disk (mostly PyTorch).
The repository ships the code and the trained model (`artifacts/model`, 2.3 MB).
The perfume data is third-party and is not redistributed here, so the first
set-up downloads it and builds the catalog locally:

```sh
git clone <this repository> scentvec
cd scentvec
python3 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/python scripts/fetch_raw_data.py     # 36 MB from pinned sources, checksummed
sh scripts/rebuild_data.sh                       # clean, link, build chemistry table
./.venv/bin/scentvec build-index                 # embed 44,658 perfumes -> artifacts/index
./.venv/bin/scentvec serve                       # then open http://127.0.0.1:8765/
```

Command line:

```sh
./.venv/bin/scentvec lookup "bleu de chanel"
./.venv/bin/scentvec similar "creed aventus" -k 8
./.venv/bin/scentvec search "smoky vanilla and tobacco, not too sweet, no patchouli"
./.venv/bin/scentvec search --accords "Fresh,Citrus" --avoid musk --gender Male --year-min 2015
./.venv/bin/scentvec blend "baccarat rouge 540" "tom ford tobacco vanille"
./.venv/bin/scentvec embed --top bergamot,lemon --middle rose --base musk,cedarwood
./.venv/bin/scentvec info
```

Every query command accepts `--json`, `-k`, `--max-per-brand` and the filters
`--gender --year-min --year-max --brand --exclude-brand --family --min-ratings
--must --avoid`. `similar` hides other editions of the same perfume line unless
`--keep-same-line` is given. Each start-up takes a few seconds to load the
catalog; the server loads it once.

Dependencies: numpy (<2), pandas, pyarrow, scipy, scikit-learn, torch (CPU is
enough). The server and tests use only the standard library.

To run from somewhere other than the source tree, set `SCENTVEC_HOME` to the
folder that contains `artifacts/` and `data/`, or pass `--model-dir` and
`--index-dir`.

## Python API

```python
from scentvec.engine import ScentEngine

engine = ScentEngine.load()

aventus = engine.lookup("creed aventus", limit=1)[0]
for row in engine.similar(aventus["item_id"], k=5, max_per_brand=1)["results"]:
    print(row["score"], row["brand"], row["name"])

found = engine.search(text="fresh citrus for summer, no musk", k=5)

vector = engine.encoder.embed(
    top=["bergamot", "lemon"],
    middle=["rose"],
    base=["musk", "cedarwood"],
)                                   # numpy array, shape (128,), unit length
```

A runnable version is in `examples/quickstart.py`. If you only need vectors and
not the catalog, load the encoder alone:

```python
from scentvec.encoder import NoteEncoder

encoder = NoteEncoder.load("artifacts/model")
vector = encoder.embed(notes=["vanilla", "tobacco", "tonka bean"])
scores = encoder.head_scores(vector)   # accords, family, subfamily
```

Other engine methods: `get`, `describe`, `explain(a, b)`, `suggest_notes`,
`stats`. Catalog rows are identified by `item_id`.

## HTTP API

`scentvec serve [--host 127.0.0.1] [--port 8765]`

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | web app |
| GET | `/api/health` | catalog and model summary |
| GET | `/api/lookup?q=` | find perfumes by name |
| GET | `/api/notes?q=` | note autocomplete |
| GET | `/api/perfume?id=` | one catalog entry |
| GET | `/api/explain?a=&b=` | what two perfumes share |
| POST | `/api/similar` | `{"id": 3435, "k": 10, "filters": {...}}` |
| POST | `/api/search` | `{"text": "...", "notes": [...], "like": [...], "k": 10}` |
| POST | `/api/embed` | `{"top": [...], "middle": [...], "base": [...], "notes": [...]}` |

Errors are JSON with an `error` field: 400 for bad input, 404 unknown route,
405 wrong method, 413 for bodies over 64 KB. The server binds to localhost by
default and has no authentication, so put it behind a proxy before exposing it.

## Rebuild from raw data

`scripts/fetch_raw_data.py` downloads the raw tables into `data/raw` from pinned
revisions and checks their SHA-256 (see MODEL_CARD.md for sources and
licences). To retrain from them:

```sh
sh scripts/rebuild_data.sh                                   # clean, link, build chemistry table
./.venv/bin/scentvec train --run-name final --epochs 15 --n-layers 1
./.venv/bin/scentvec evaluate --run final --split test       # writes reports/eval_final_test.md
PYTHONPATH=src ./.venv/bin/python scripts/paired_bootstrap.py --run final --split test
./.venv/bin/scentvec package --run final                     # -> artifacts/model
./.venv/bin/scentvec build-index                             # -> artifacts/index
```

Training takes about 10 minutes on a 4-core laptop CPU. Ablations:
`sh scripts/hone_ablations.sh`, then `scripts/ablation_table.py`.

## Tests

```sh
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -t .
```

84 tests cover text normalisation, data cleaning, query parsing, the encoder,
the engine and the HTTP server.

## Layout

```
src/scentvec/      library, CLI, server, web app (web/index.html)
scripts/           raw-data fetch, data rebuild, ablations, bootstrap significance
tests/             unit tests
examples/          quickstart.py
data/raw/          source tables, fetched by scripts/fetch_raw_data.py (not in git)
data/processed/    cleaned records, cross-database pairs, chemistry table (not in git)
artifacts/model/   packaged model bundle (2.3 MB)
artifacts/index/   catalog and its embeddings (32 MB, built locally, not in git)
reports/           evaluation, significance, ablations, before/after snapshots
```

## Licence note

No licence has been chosen for the code. The training data is scraped,
community-edited perfume data with unclear or research-only terms, which is
why this repository does not contain it and fetches it from its original
hosts instead. The packaged model weights were trained on that data. Check
MODEL_CARD.md before redistributing the catalog or using it commercially.
