"""HTTP API: handler logic and one real round trip on an ephemeral port."""

from __future__ import annotations

import http.client
import json
import threading
import unittest

from tests import helpers
from scentvec.server import MAX_BODY_BYTES, Api, create_server


class DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = Api(helpers.build_tiny_engine())

    def call(self, method, path, argument=None):
        status, body = self.api.dispatch(method, path, argument or {})
        json.dumps(body)  # every body must be JSON-serialisable
        return int(status), body

    def test_health(self):
        status, body = self.call("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["perfumes"], 8)

    def test_lookup_and_perfume(self):
        status, body = self.call("GET", "/api/lookup", {"q": "creed aventus", "limit": "2"})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["item_id"], 0)
        self.assertEqual(len(body["results"]), 2)
        status, body = self.call("GET", "/api/perfume", {"id": "4"})
        self.assertEqual(status, 200)
        self.assertEqual(body["name"], "Baccarat Rouge 540")

    def test_bad_requests(self):
        cases = [
            ("GET", "/api/perfume", {}),
            ("GET", "/api/perfume", {"id": "abc"}),
            ("GET", "/api/perfume", {"id": "99"}),
            ("GET", "/api/lookup", {"q": "x", "limit": "many"}),
            ("GET", "/api/explain", {"a": "0"}),
            ("POST", "/api/similar", {}),
            ("POST", "/api/similar", {"id": 0, "surprise": 1}),
            ("POST", "/api/similar", {"id": 0, "filters": {"genders": 5}}),
            ("POST", "/api/similar", {"id": 0, "filters": "Male"}),
            ("POST", "/api/search", {}),
            ("POST", "/api/search", {"text": 5}),
            ("POST", "/api/search", {"text": "x" * 501}),
            ("POST", "/api/search", {"notes": "rose", "like": "0"}),
            ("POST", "/api/search", {"like": [4, 5], "like_weights": {"a": 1}}),
            ("POST", "/api/search", {"accords": 3}),
            ("POST", "/api/embed", {}),
            ("POST", "/api/embed", {"notes": [1]}),
        ]
        for method, path, argument in cases:
            with self.subTest(path=path, argument=argument):
                status, body = self.call(method, path, argument)
                self.assertEqual(status, 400)
                self.assertIn("error", body)

    def test_unknown_route_and_wrong_method(self):
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)
        self.assertEqual(self.call("GET", "/api/search")[0], 405)
        self.assertEqual(self.call("POST", "/api/health")[0], 405)

    def test_similar_search_embed(self):
        status, body = self.call("POST", "/api/similar", {"id": 0, "k": 3, "filters": {"gender": "Male"}})
        self.assertEqual(status, 200)
        self.assertEqual([entry["item_id"] for entry in body["results"]], [3])
        status, body = self.call("POST", "/api/search", {"notes": ["tobacco", "vanilla"], "k": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["item_id"], 5)
        status, body = self.call("POST", "/api/embed", {"top": ["bergamot"], "base": ["musk"], "k": 2})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["vector"]), 16)
        self.assertEqual(len(body["nearest"]), 2)

    def test_internal_errors_are_hidden(self):
        broken = Api(engine=None)
        with self.assertLogs("scentvec.server", level="ERROR"):
            status, body = broken.dispatch("GET", "/api/health", {})
        self.assertEqual(int(status), 500)
        self.assertEqual(body, {"error": "internal error"})


class HttpRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(helpers.build_tiny_engine(), port=0, quiet=True)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def post_json(self, path, payload):
        raw = json.dumps(payload).encode("utf-8")
        status, _, body = self.request("POST", path, raw, {"Content-Type": "application/json"})
        return status, json.loads(body)

    def test_web_page(self):
        status, content_type, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/html"))
        self.assertIn(b"<html", body.lower())

    def test_get_with_query_string(self):
        status, content_type, body = self.request("GET", "/api/lookup?q=tom%20ford")
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("application/json"))
        self.assertEqual(json.loads(body)["results"][0]["name"], "Tobacco Vanille")

    def test_non_ascii_round_trip(self):
        status, _, body = self.request("GET", "/api/perfume?id=6")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode("utf-8"))["name"], "N°5")

    def test_post_search(self):
        status, body = self.post_json("/api/search", {"text": "like baccarat rouge 540", "k": 3})
        self.assertEqual(status, 200)
        self.assertEqual(body["query"]["like"][0]["item_id"], 4)
        self.assertEqual(len(body["results"]), 3)

    def test_bad_json_and_non_object(self):
        status, _, body = self.request("POST", "/api/search", b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))
        status, _, _ = self.request("POST", "/api/search", b"[1, 2]")
        self.assertEqual(status, 400)

    def test_oversized_body(self):
        headers = {"Content-Length": str(MAX_BODY_BYTES + 1)}
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.putrequest("POST", "/api/search")
            connection.putheader("Content-Length", headers["Content-Length"])
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            connection.close()

    def test_not_found_and_method_not_allowed(self):
        self.assertEqual(self.request("GET", "/api/nope")[0], 404)
        self.assertEqual(self.request("GET", "/api/similar")[0], 405)
        self.assertEqual(self.request("GET", "/etc/passwd")[0], 404)


if __name__ == "__main__":
    unittest.main()
