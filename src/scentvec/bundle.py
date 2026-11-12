"""Package a training run as the model bundle the product loads.

A bundle is a self-contained directory:

    model.pt         encoder weights
    config.json      architecture and training settings
    vocab.json       note, word, accord and family vocabularies
    chem_table.npz   odor-chemistry profiles for notes
    bundle.json      version, source run and validation metrics
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import __version__
from .chem import CHEM_TABLE_FILE
from .data import PROCESSED_DIR

BUNDLE_FILES = ["model.pt", "config.json", "vocab.json"]


def package_run(run_dir: Path, model_dir: Path, processed_dir: Path = PROCESSED_DIR) -> dict:
    run_dir = Path(run_dir)
    model_dir = Path(model_dir)
    for name in BUNDLE_FILES:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir / name} is missing; has the run finished an epoch?")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        shutil.copy2(run_dir / name, model_dir / name)
    shutil.copy2(processed_dir / CHEM_TABLE_FILE, model_dir / CHEM_TABLE_FILE)

    validation = {}
    metrics_path = run_dir / "val_metrics.json"
    if metrics_path.exists():
        validation = json.loads(metrics_path.read_text())
    manifest = {
        "scentvec_version": __version__,
        "source_run": run_dir.name,
        "packaged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_epoch": validation.get("epoch"),
        "validation_score": validation.get("score"),
        "validation_metrics": validation.get("metrics"),
    }
    (model_dir / "bundle.json").write_text(json.dumps(manifest, indent=2))
    return manifest
