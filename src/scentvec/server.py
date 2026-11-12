"""Local HTTP service: a JSON API plus the single-page web app.

Standard library only.  Binds to localhost by default.

    GET  /                      web app
    GET  /api/health            status, version, catalog size
    GET  /api/lookup?q=         find perfumes by name
    GET  /api/notes?q=          note autocomplete
    GET  /api/perfume?id=       one catalog entry
    GET  /api/explain?a=&b=     what two perfumes share
    POST /api/similar           {"id": 12, "k": 10, "filters": {...}}
    POST /api/search            {"text": "...", "notes": [...], "like": [...], ...}
    POST /api/embed             {"top": [...], "middle": [...], "base": [...], "notes": [...]}
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import __version__
from .engine import EngineError, Filters, ScentEngine

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_LENGTH = 500
logger = logging.getLogger("scentvec.server")

SEARCH_FIELDS = {
    "text", "top", "middle", "base", "notes", "accords", "avoid_notes", "avoid_accords",
    "like", "unlike", "like_weights", "filters", "k", "max_per_brand",
}
SIMILAR_FIELDS = {"id", "k", "filters", "exclude_same_line", "max_per_brand"}
EMBED_FIELDS = {"top", "middle", "base", "notes", "k", "include_vector"}


def _integer(value, name: str, default: Optional[int] = None) -> int:
    if value is None:
        if default is None:
            raise EngineError(f"'{name}' is required")
        return default
    if isinstance(value, bool):
        raise EngineError(f"'{name}' must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise EngineError(f"'{name}' must be an integer")


def _check_fields(payload: dict, allowed: set) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise EngineError(f"unknown field(s): {', '.join(sorted(unknown))}")


def _id_list(values, name: str) -> list:
    if values is None:
        return []
    if not isinstance(values, list):
        raise EngineError(f"'{name}' must be a list of perfume ids")
    return [_integer(value, name) for value in values]


class Api:
    """Request handlers, independent of the HTTP plumbing so they can be tested."""

    def __init__(self, engine: ScentEngine) -> None:
        self.engine = engine

    def health(self, query: Dict[str, str]) -> dict:
        return {"status": "ok", "version": __version__, **self.engine.stats()}

    def lookup(self, query: Dict[str, str]) -> dict:
        text = query.get("q", "")[:MAX_TEXT_LENGTH]
        limit = _integer(query.get("limit"), "limit", 10)
        return {"results": self.engine.lookup(text, limit)}

    def notes(self, query: Dict[str, str]) -> dict:
        return {"results": self.engine.suggest_notes(query.get("q", "")[:80])}

    def perfume(self, query: Dict[str, str]) -> dict:
        return self.engine.get(_integer(query.get("id"), "id"))

    def explain(self, query: Dict[str, str]) -> dict:
        return self.engine.explain(_integer(query.get("a"), "a"), _integer(query.get("b"), "b"))

    def similar(self, payload: dict) -> dict:
        _check_fields(payload, SIMILAR_FIELDS)
        return self.engine.similar(
            _integer(payload.get("id"), "id"),
            k=_integer(payload.get("k"), "k", 10),
            filters=Filters.from_json(payload.get("filters")),
            exclude_same_line=bool(payload.get("exclude_same_line", True)),
            max_per_brand=_integer(payload.get("max_per_brand"), "max_per_brand", 0),
        )

    def search(self, payload: dict) -> dict:
        _check_fields(payload, SEARCH_FIELDS)
        text = payload.get("text")
        if text is not None and not isinstance(text, str):
            raise EngineError("'text' must be a string")
        if text and len(text) > MAX_TEXT_LENGTH:
            raise EngineError(f"'text' is limited to {MAX_TEXT_LENGTH} characters")
        return self.engine.search(
            text=text,
            top=payload.get("top"),
            middle=payload.get("middle"),
            base=payload.get("base"),
            notes=payload.get("notes"),
            accords=payload.get("accords"),
            avoid_notes=payload.get("avoid_notes"),
            avoid_accords=payload.get("avoid_accords"),
            like=_id_list(payload.get("like"), "like"),
            unlike=_id_list(payload.get("unlike"), "unlike"),
            like_weights=payload.get("like_weights"),
            filters=Filters.from_json(payload.get("filters")),
            k=_integer(payload.get("k"), "k", 10),
            max_per_brand=_integer(payload.get("max_per_brand"), "max_per_brand", 0),
        )

    def embed(self, payload: dict) -> dict:
        _check_fields(payload, EMBED_FIELDS)
        described = self.engine.describe(
            top=self.engine._canonical_list(payload.get("top"), "top"),
            middle=self.engine._canonical_list(payload.get("middle"), "middle"),
            base=self.engine._canonical_list(payload.get("base"), "base"),
            notes=self.engine._canonical_list(payload.get("notes"), "notes"),
            include_vector=bool(payload.get("include_vector", True)),
        )
        k = _integer(payload.get("k"), "k", 0)
        if k > 0:
            nearest = self.engine.search(
                top=payload.get("top"),
                middle=payload.get("middle"),
                base=payload.get("base"),
                notes=payload.get("notes"),
                k=k,
            )
            described["nearest"] = nearest["results"]
        return described

    GET_ROUTES = {
        "/api/health": "health",
        "/api/lookup": "lookup",
        "/api/notes": "notes",
        "/api/perfume": "perfume",
        "/api/explain": "explain",
    }
    POST_ROUTES = {"/api/similar": "similar", "/api/search": "search", "/api/embed": "embed"}

    def dispatch(self, method: str, path: str, argument: dict) -> Tuple[int, dict]:
        """Run one API call: (HTTP status, JSON-serialisable body)."""
        routes = self.GET_ROUTES if method == "GET" else self.POST_ROUTES
        name = routes.get(path)
        if name is None:
            other = self.POST_ROUTES if method == "GET" else self.GET_ROUTES
            if path in other:
                return HTTPStatus.METHOD_NOT_ALLOWED, {"error": f"{path} does not accept {method}"}
            return HTTPStatus.NOT_FOUND, {"error": f"no such endpoint: {path}"}
        handler: Callable[[dict], dict] = getattr(self, name)
        try:
            return HTTPStatus.OK, handler(argument)
        except EngineError as error:
            return HTTPStatus.BAD_REQUEST, {"error": str(error)}
        except Exception:  # noqa: BLE001 - last line of defence for a server
            logger.error("unhandled error in %s\n%s", path, traceback.format_exc())
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"}


def make_handler(api: Api, quiet: bool = False):
    index_html = (WEB_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"scentvec/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, format_string: str, *args) -> None:
            if not quiet:
                logger.info("%s %s", self.address_string(), format_string % args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - name fixed by http.server
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, index_html, "text/html; charset=utf-8")
                return
            query = {name: values[-1] for name, values in parse_qs(url.query).items()}
            status, payload = api.dispatch("GET", url.path, query)
            self._send_json(status, payload)

        do_HEAD = do_GET  # noqa: N815

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self.close_connection = True
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body must be valid JSON"})
                return
            if not isinstance(payload, dict):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body must be a JSON object"})
                return
            status, body = api.dispatch("POST", url.path, payload)
            self._send_json(status, body)

    return Handler


def create_server(
    engine: ScentEngine, host: str = "127.0.0.1", port: int = 8765, quiet: bool = False
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(Api(engine), quiet))
    server.daemon_threads = True
    return server


def serve(host: str = "127.0.0.1", port: int = 8765, model_dir=None, index_dir=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    arguments = {}
    if model_dir is not None:
        arguments["model_dir"] = model_dir
    if index_dir is not None:
        arguments["index_dir"] = index_dir
    engine = ScentEngine.load(**arguments)
    server = create_server(engine, host, port)
    print(f"scentvec {__version__}: {len(engine):,} perfumes", flush=True)
    print(f"open http://{host}:{server.server_address[1]}/   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
