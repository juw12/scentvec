"""Command line interface.

Using the product:

    scentvec similar "creed aventus"
    scentvec search "smoky vanilla and tobacco, not too sweet, no patchouli"
    scentvec blend "baccarat rouge 540" "tom ford tobacco vanille"
    scentvec embed --top bergamot,lemon --middle rose --base musk,cedarwood
    scentvec serve

Rebuilding it from the raw data:

    scentvec build-data && scentvec build-chem
    scentvec train --run-name final
    scentvec evaluate --run-dir artifacts/runs/final --split test
    scentvec package --run final && scentvec build-index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__

PIPELINE_COMMANDS = {
    "build-data": ("scentvec.data", "clean and link the raw perfume tables"),
    "build-chem": ("scentvec.chem", "build the note odor-chemistry table"),
    "train": ("scentvec.train", "train the encoder"),
    "evaluate": ("scentvec.evaluate", "evaluate a run against the baselines"),
    "build-index": ("scentvec.catalog", "embed the catalog with the packaged model"),
}


def _split_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("filters")
    group.add_argument("--gender", help="comma list of Male, Female, Unisex")
    group.add_argument("--year-min", type=int)
    group.add_argument("--year-max", type=int)
    group.add_argument("--brand", help="only these brands (comma list)")
    group.add_argument("--exclude-brand", help="skip these brands (comma list)")
    group.add_argument("--family", help="olfactive family (comma list)")
    group.add_argument("--min-ratings", type=int, default=0, help="minimum number of ratings")
    group.add_argument("--must", help="notes that must be listed (comma list)")
    group.add_argument("--avoid", help="notes that must not be listed (comma list)")
    parser.add_argument("-k", type=int, default=10, help="number of results")
    parser.add_argument("--max-per-brand", type=int, default=0, help="0 = no limit")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")


def _filters_from(arguments: argparse.Namespace):
    from .engine import Filters

    return Filters(
        genders=_split_list(arguments.gender),
        year_min=arguments.year_min,
        year_max=arguments.year_max,
        brands=_split_list(arguments.brand),
        exclude_brands=_split_list(arguments.exclude_brand),
        families=_split_list(arguments.family),
        min_rating_count=arguments.min_ratings,
        must_notes=_split_list(arguments.must),
        exclude_notes=_split_list(arguments.avoid),
    )


def _title(entry: dict) -> str:
    parts = [f"{entry['brand']} - {entry['name']}"]
    details = [str(entry[name]) for name in ("year", "concentration", "gender") if entry.get(name)]
    if details:
        parts.append("(" + ", ".join(details) + ")")
    return " ".join(parts)


def _notes_line(entry: dict) -> str:
    sections = []
    for label, column in [
        ("top", "notes_top"), ("heart", "notes_middle"), ("base", "notes_base"), ("notes", "notes_other"),
    ]:
        if entry.get(column):
            sections.append(f"{label}: {', '.join(entry[column][:8])}")
    return " | ".join(sections)


def _print_results(results: List[dict]) -> None:
    if not results:
        print("  no perfumes match")
        return
    for rank, entry in enumerate(results, start=1):
        score = f"{entry['score']:.3f}  " if "score" in entry else ""
        print(f"{rank:2d}. {score}{_title(entry)}   [id {entry['item_id']}]")
        accords = entry["accords"]
        flag = "" if accords["source"] == "voted" else " (predicted)"
        print(f"      accords{flag}: {', '.join(accords['values'])}")
        print(f"      {_notes_line(entry)}")
        if entry.get("shared_notes"):
            print(f"      shared: {', '.join(entry['shared_notes'][:10])}")


def _resolve(engine, reference: str) -> int:
    """A perfume given as a catalog id or as a name."""
    from .engine import EngineError

    if reference.isdigit():
        return int(reference)
    found = engine.lookup(reference, limit=5)
    if not found:
        raise EngineError(f"no perfume found for '{reference}'")
    if not found[0]["matched_all_words"]:
        suggestions = "; ".join(_title(entry) for entry in found[:3])
        raise EngineError(f"no perfume matches all of '{reference}'. Closest: {suggestions}")
    return found[0]["item_id"]


def _load_engine(arguments: argparse.Namespace):
    from .catalog import INDEX_DIR, MODEL_DIR
    from .engine import ScentEngine

    return ScentEngine.load(arguments.model_dir or MODEL_DIR, arguments.index_dir or INDEX_DIR)


def _command_lookup(arguments: argparse.Namespace) -> None:
    engine = _load_engine(arguments)
    results = engine.lookup(arguments.name, arguments.k)
    if arguments.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _print_results(results)


def _command_similar(arguments: argparse.Namespace) -> None:
    engine = _load_engine(arguments)
    item_id = _resolve(engine, arguments.perfume)
    response = engine.similar(
        item_id,
        k=arguments.k,
        filters=_filters_from(arguments),
        exclude_same_line=not arguments.keep_same_line,
        max_per_brand=arguments.max_per_brand,
    )
    if arguments.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return
    query = response["query"]
    print(f"Similar to {_title(query)}   [id {query['item_id']}]")
    print(f"      {_notes_line(query)}\n")
    _print_results(response["results"])


def _print_search(response: dict) -> None:
    query = response["query"]
    understood = []
    if query["like"]:
        understood.append("like " + " + ".join(_title(entry) for entry in query["like"]))
    if query["notes"]:
        understood.append("notes: " + ", ".join(entry["note"] for entry in query["notes"]))
    if query["accords"]:
        understood.append("accords: " + ", ".join(query["accords"]))
    if query["avoid_notes"]:
        understood.append("without notes: " + ", ".join(query["avoid_notes"]))
    if query["avoid_accords"]:
        understood.append("less: " + ", ".join(query["avoid_accords"]))
    if query["filters"]:
        understood.append("filters: " + json.dumps(query["filters"]))
    print("Understood  " + "\n            ".join(understood))
    if query["unrecognised"]:
        print("Ignored     " + ", ".join(query["unrecognised"]))
    print(f"Candidates  {response['candidates']:,}\n")
    _print_results(response["results"])


def _command_search(arguments: argparse.Namespace) -> None:
    engine = _load_engine(arguments)
    response = engine.search(
        text=" ".join(arguments.text) or None,
        top=_split_list(arguments.top),
        middle=_split_list(arguments.middle),
        base=_split_list(arguments.base),
        notes=_split_list(arguments.notes),
        accords=_split_list(arguments.accords),
        avoid_accords=_split_list(arguments.avoid_accords),
        like=[_resolve(engine, reference) for reference in arguments.like or []],
        unlike=[_resolve(engine, reference) for reference in arguments.unlike or []],
        filters=_filters_from(arguments),
        k=arguments.k,
        max_per_brand=arguments.max_per_brand,
    )
    if arguments.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        _print_search(response)


def _command_blend(arguments: argparse.Namespace) -> None:
    from .engine import EngineError

    engine = _load_engine(arguments)
    item_ids = [_resolve(engine, reference) for reference in arguments.perfumes]
    weights = None
    if arguments.weights:
        try:
            weights = [float(value) for value in _split_list(arguments.weights)]
        except ValueError:
            raise EngineError("--weights must be a comma list of numbers")
    response = engine.search(
        like=item_ids,
        like_weights=weights,
        filters=_filters_from(arguments),
        k=arguments.k,
        max_per_brand=arguments.max_per_brand,
    )
    if arguments.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        _print_search(response)


def _command_embed(arguments: argparse.Namespace) -> None:
    engine = _load_engine(arguments)
    described = engine.describe(
        top=_split_list(arguments.top),
        middle=_split_list(arguments.middle),
        base=_split_list(arguments.base),
        notes=_split_list(arguments.notes),
    )
    if arguments.json:
        print(json.dumps(described, ensure_ascii=False))
        return
    for note in described["notes"]:
        print(f"  {note['position']:11s} {note['note']:28s} seen as: {note['recognised_as']}")
    print("accords:   " + ", ".join(f"{a['accord']} {a['probability']:.2f}" for a in described["accords"]))
    print("family:    " + ", ".join(f"{f['family']} {f['probability']:.2f}" for f in described["family"]))
    print("subfamily: " + ", ".join(f"{s['subfamily']} {s['probability']:.2f}" for s in described["subfamily"]))
    preview = ", ".join(f"{value:+.4f}" for value in described["vector"][:8])
    print(f"vector:    {described['dim']} dims, unit length  [{preview}, ...]   (--json for all)")


def _command_info(arguments: argparse.Namespace) -> None:
    from .catalog import MODEL_DIR

    engine = _load_engine(arguments)
    info = engine.stats()
    manifest_path = Path(arguments.model_dir or MODEL_DIR) / "bundle.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        info["model"] = {
            key: manifest.get(key)
            for key in ("scentvec_version", "source_run", "packaged_at", "selected_epoch")
        }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def _command_package(arguments: argparse.Namespace) -> None:
    from .bundle import package_run
    from .catalog import MODEL_DIR
    from .train import resolve_run_dir

    run_dir = resolve_run_dir(arguments.run)
    manifest = package_run(run_dir, arguments.model_dir or MODEL_DIR)
    print(
        f"packaged run '{manifest['source_run']}' (epoch {manifest['selected_epoch']}) "
        f"into {arguments.model_dir or MODEL_DIR}"
    )


def _command_serve(arguments: argparse.Namespace) -> None:
    from .server import serve

    serve(arguments.host, arguments.port, arguments.model_dir, arguments.index_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scentvec",
        description="Perfume embedding model: find similar perfumes, search by description.",
    )
    parser.add_argument("--version", action="version", version=f"scentvec {__version__}")
    parser.add_argument("--model-dir", type=Path, default=None, help="model bundle directory")
    parser.add_argument("--index-dir", type=Path, default=None, help="catalog index directory")
    commands = parser.add_subparsers(dest="command", metavar="command")

    lookup = commands.add_parser("lookup", help="find a perfume by name")
    lookup.add_argument("name")
    lookup.add_argument("-k", type=int, default=10)
    lookup.add_argument("--json", action="store_true")
    lookup.set_defaults(handler=_command_lookup)

    similar = commands.add_parser("similar", help="perfumes that smell like a given one")
    similar.add_argument("perfume", help="name or catalog id")
    similar.add_argument("--keep-same-line", action="store_true", help="keep other versions of it")
    _add_filter_arguments(similar)
    similar.set_defaults(handler=_command_similar)

    search = commands.add_parser("search", help="search by description, notes and accords")
    search.add_argument("text", nargs="*", help="free text, e.g. 'fresh citrus, no musk, for men'")
    search.add_argument("--top", help="top notes (comma list)")
    search.add_argument("--middle", help="heart notes (comma list)")
    search.add_argument("--base", help="base notes (comma list)")
    search.add_argument("--notes", help="notes of any position (comma list)")
    search.add_argument("--accords", help="wanted accords (comma list)")
    search.add_argument("--avoid-accords", help="unwanted accords (comma list)")
    search.add_argument("--like", action="append", help="reference perfume (repeatable)")
    search.add_argument("--unlike", action="append", help="perfume to move away from (repeatable)")
    _add_filter_arguments(search)
    search.set_defaults(handler=_command_search)

    blend = commands.add_parser("blend", help="perfumes between two or more given ones")
    blend.add_argument("perfumes", nargs="+", help="names or catalog ids")
    blend.add_argument("--weights", help="comma list, one weight per perfume")
    _add_filter_arguments(blend)
    blend.set_defaults(handler=_command_blend)

    embed = commands.add_parser("embed", help="embed a list of notes")
    embed.add_argument("--top")
    embed.add_argument("--middle")
    embed.add_argument("--base")
    embed.add_argument("--notes")
    embed.add_argument("--json", action="store_true", help="print the full vector as JSON")
    embed.set_defaults(handler=_command_embed)

    info = commands.add_parser("info", help="catalog and model summary")
    info.set_defaults(handler=_command_info)

    serve = commands.add_parser("serve", help="start the web app and JSON API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(handler=_command_serve)

    package = commands.add_parser("package", help="turn a training run into the model bundle")
    package.add_argument("--run", required=True, help="run name or directory")
    package.set_defaults(handler=_command_package)

    for name, (_, description) in PIPELINE_COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Pipeline commands own their arguments; hand everything after the name over.
    for position, word in enumerate(argv):
        if word in PIPELINE_COMMANDS:
            import importlib

            module = importlib.import_module(PIPELINE_COMMANDS[word][0])
            module.main(argv[position + 1 :])
            return 0
        if not word.startswith("-"):
            break

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not getattr(arguments, "handler", None):
        parser.print_help()
        return 2

    from .engine import EngineError

    try:
        arguments.handler(arguments)
    except (EngineError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
