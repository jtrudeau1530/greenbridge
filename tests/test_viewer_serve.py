"""Tests for viewer/serve.py: the loopback Host check, /world/ path resolution and the HTTP layer.

    python -I tests/test_viewer_serve.py

Everything runs on a temporary world folder and a loopback server: no game data, no browser, no network beyond
127.0.0.1, standard library only.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
if str(VIEWER) not in sys.path:
    sys.path.insert(0, str(VIEWER))

import serve  # noqa: E402


class HostAllowedTest(unittest.TestCase):
    """Only a Host header naming this machine may be answered (DNS rebinding)."""

    def test_loopback_hosts_are_allowed(self):
        for host in ("127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:1", "LOCALHOST:8765",
                     "localhost.", "[::1]", "[::1]:8765"):
            with self.subTest(host=host):
                self.assertTrue(serve.host_allowed(host))

    def test_other_hosts_are_refused(self):
        for host in ("", "evil.example", "evil.example:8765", "127.0.0.1.evil.example",
                     "0.0.0.0:8765", "::1", "localhost.evil.example:8765"):
            with self.subTest(host=host):
                self.assertFalse(serve.host_allowed(host))

    def test_a_request_without_a_host_header_is_allowed(self):
        # Browsers always send Host; a missing one is a local tool, not a web page.
        self.assertTrue(serve.host_allowed(None))


class SafeWorldPathTest(unittest.TestCase):
    """safe_world_path resolves /world/... inside the world folder or refuses it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.world = self.root / "world"
        (self.world / "sub").mkdir(parents=True)
        (self.world / "ok.bin").write_bytes(b"world-bytes")
        (self.world / "sub" / "inner.bin").write_bytes(b"inner")
        # A file the world must never reach, next to the world folder.
        (self.root / "secret.txt").write_text("secret", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def resolve(self, request_path):
        return serve.safe_world_path(str(self.world), request_path)

    def test_a_world_file_resolves_to_its_real_path(self):
        resolved = self.resolve("/world/ok.bin")
        self.assertIsNotNone(resolved)
        self.assertEqual(os.path.normcase(os.path.realpath(resolved)),
                         os.path.normcase(os.path.realpath(self.world / "ok.bin")))

    def test_a_nested_world_file_resolves(self):
        self.assertIsNotNone(self.resolve("/world/sub/inner.bin"))

    def test_query_and_fragment_are_not_part_of_the_path(self):
        self.assertIsNotNone(self.resolve("/world/ok.bin?cache=1"))
        self.assertIsNotNone(self.resolve("/world/ok.bin#top"))

    def test_requests_outside_world_are_refused(self):
        for path in ("/ok.bin", "/", "/world", "/world/../secret.txt", "/world/sub/../../secret.txt",
                     "/world/%2e%2e/secret.txt", "/world/..%2fsecret.txt", "/world/..\\secret.txt",
                     "/world//ok.bin", "/world/./ok.bin", "/world/", "/world/sub/", "/world/sub"):
            with self.subTest(path=path):
                self.assertIsNone(self.resolve(path))

    def test_nul_bytes_and_drive_letters_are_refused(self):
        for path in ("/world/ok%00.bin", "/world/C:secret.txt", "/world/sub/inner.bin:stream"):
            with self.subTest(path=path):
                self.assertIsNone(self.resolve(path))

    def test_windows_reserved_device_names_are_refused(self):
        for name in ("CON", "con.txt", "NUL", "COM1", "LPT9.log", "aux."):
            with self.subTest(name=name):
                self.assertIsNone(self.resolve("/world/" + name))

    def test_a_missing_world_or_bad_argument_is_refused(self):
        self.assertIsNone(serve.safe_world_path("", "/world/ok.bin"))
        self.assertIsNone(serve.safe_world_path(str(self.world), None))
        self.assertIsNone(serve.safe_world_path(str(self.world), b"/world/ok.bin"))

    def test_a_symlink_out_of_the_world_is_refused(self):
        link = self.world / "escape.bin"
        try:
            os.symlink(self.root / "secret.txt", link)
        except (OSError, NotImplementedError) as e:      # Windows without the privilege
            self.skipTest(f"cannot create a symlink here: {e}")
        self.assertIsNone(self.resolve("/world/escape.bin"))


def raw_request(port, target, host="127.0.0.1", method="GET"):
    """Send a request without letting a client library normalise the target path."""
    lines = [f"{method} {target} HTTP/1.1"]
    if host is not None:
        lines.append(f"Host: {host}")
    lines += ["Connection: close", "", ""]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall("\r\n".join(lines).encode("latin-1"))
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, head.decode("latin-1"), body


class HandlerTest(unittest.TestCase):
    """/world/ files are served with the guards in front of them; a foreign Host gets 403."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        world = root / "world"
        (world / "sub").mkdir(parents=True)
        (world / "ok.bin").write_bytes(b"world-bytes")
        (world / "model.glb").write_bytes(b"glTF")
        (root / "site").mkdir()
        (root / "site" / "index.html").write_text("<!doctype html><title>viewer</title>", encoding="utf-8")
        (root / "secret.txt").write_text("secret", encoding="utf-8")

        class QuietHandler(serve.Handler):
            def log_message(self, *args):
                pass

        QuietHandler.world = str(world)      # a class body cannot see the enclosing function's locals

        cls.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root / "site")))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)
        cls._tmp.cleanup()

    def test_a_world_file_is_served_without_caching(self):
        status, head, body = raw_request(self.port, "/world/ok.bin", host=f"127.0.0.1:{self.port}")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"world-bytes")
        self.assertIn("cache-control: no-store", head.lower())

    def test_the_glb_mime_type_is_registered(self):
        status, head, _ = raw_request(self.port, "/world/model.glb", host=f"localhost:{self.port}")
        self.assertEqual(status, 200)
        self.assertIn("content-type: model/gltf-binary", head.lower())

    def test_head_and_query_strings_work(self):
        status, _, body = raw_request(self.port, "/world/ok.bin?cache=1", host="127.0.0.1", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_a_missing_host_header_is_served(self):
        status, _, body = raw_request(self.port, "/world/ok.bin", host=None)
        self.assertEqual(status, 200)
        self.assertEqual(body, b"world-bytes")

    def test_a_foreign_host_gets_403(self):
        for target in ("/world/ok.bin", "/index.html"):
            with self.subTest(target=target):
                status, head, _ = raw_request(self.port, target, host="evil.example:8765")
                self.assertEqual(status, 403)
                self.assertIn("cache-control: no-store", head.lower())

    def test_traversal_attempts_get_403(self):
        for target in ("/world/../secret.txt", "/world/%2e%2e/secret.txt",
                       "/world/sub/../../secret.txt", "/world/..%5csecret.txt"):
            with self.subTest(target=target):
                status, _, body = raw_request(self.port, target, host=f"127.0.0.1:{self.port}")
                self.assertEqual(status, 403)
                self.assertNotIn(b"secret", body)

    def test_the_viewer_page_itself_is_served(self):
        status, _, body = raw_request(self.port, "/index.html", host=f"127.0.0.1:{self.port}")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>viewer</title>", body)


class PortArgumentTest(unittest.TestCase):
    """--port is 0-65535; outside that it used to reach socketserver and raise OverflowError."""

    def test_valid_ports(self):
        self.assertEqual(serve.port_number("0"), 0)
        self.assertEqual(serve.port_number("8765"), 8765)
        self.assertEqual(serve.port_number("65535"), 65535)

    def test_invalid_ports_are_refused_by_argparse(self):
        for text in ("-1", "65536", "70000", "abc", ""):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    serve.port_number(text)

    def test_the_cli_reports_a_bad_port_without_a_traceback(self):
        result = subprocess.run(
            [sys.executable, "-I", str(VIEWER / "serve.py"), "--port", "70000"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 0-65535", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
