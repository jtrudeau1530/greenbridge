"""Unit tests for tools/gallery.py standalone 3D Studio & Asset Gallery server."""
import json
import io
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import gallery


class TestGalleryCategorization(unittest.TestCase):
    def test_categories(self) -> None:
        self.assertEqual(gallery.categorize_model("ped_lucia_body.glb"), "Персонажи (Characters)")
        self.assertEqual(gallery.categorize_model("veh_infernus.glb"), "Транспорт (Vehicles)")
        self.assertEqual(gallery.categorize_model("w_rifle_m4.glb"), "Оружие (Weapons)")
        self.assertEqual(gallery.categorize_model("build_wall_01.glb"), "Карта и Здания (World)")
        self.assertEqual(gallery.categorize_model("prop_chair_office.glb"), "Объекты и Пропы (Props)")
        self.assertEqual(gallery.categorize_model("tree_palm_02.glb"), "Растительность (Nature)")
        self.assertEqual(gallery.categorize_model("misc_something.glb"), "Прочее (Miscellaneous)")

    def test_folder_inference(self) -> None:
        self.assertEqual(gallery.categorize_model("asset_01.glb", "clean_vehicles"), "Транспорт (Vehicles)")
        self.assertEqual(gallery.categorize_model("model.glb", "unpacked_peds"), "Персонажи (Characters)")


class TestGalleryServer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.tmp.name, "models")
        self.textures_dir = os.path.join(self.tmp.name, "textures")
        self.viewer_dir = os.path.join(self.tmp.name, "viewer")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.textures_dir, exist_ok=True)
        os.makedirs(self.viewer_dir, exist_ok=True)

        # Create dummy assets
        with open(os.path.join(self.models_dir, "test_car.glb"), "wb") as f:
            f.write(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")

        with open(os.path.join(self.textures_dir, "test_tex.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        with open(os.path.join(self.viewer_dir, "studio.html"), "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><body><h1>Studio</h1></body></html>")

        # Configure handler
        gallery.StudioHandler.models_dir = self.models_dir
        gallery.StudioHandler.textures_dir = self.textures_dir
        gallery.StudioHandler.viewer_dir = self.viewer_dir

        # Start server on an open port
        addr = ("127.0.0.1", 0)
        self.httpd = gallery.socketserver.TCPServer(addr, gallery.StudioHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_endpoints(self) -> None:
        base = f"http://127.0.0.1:{self.port}"

        # 1. Studio HTML root
        with urllib.request.urlopen(f"{base}/", timeout=3) as r:
            self.assertEqual(r.status, 200)
            body = r.read().decode("utf-8")
            self.assertIn("<h1>Studio</h1>", body)

        # 2. Models API
        with urllib.request.urlopen(f"{base}/api/models", timeout=3) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["models"][0]["name"], "test_car.glb")
            self.assertEqual(data["models"][0]["category"], "Транспорт (Vehicles)")

        # 3. Textures API
        with urllib.request.urlopen(f"{base}/api/textures", timeout=3) as r:
            self.assertEqual(r.status, 200)
            tdata = json.loads(r.read())
            self.assertEqual(tdata["count"], 1)
            self.assertEqual(tdata["textures"][0]["name"], "test_tex.png")

        # 4. Binary serving
        with urllib.request.urlopen(f"{base}/models/test_car.glb", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"glTF"))

        with urllib.request.urlopen(f"{base}/textures/test_tex.png", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"\x89PNG"))


class TestGalleryCommandLine(unittest.TestCase):
    def test_rejects_out_of_range_ports_before_starting_server(self) -> None:
        for port in (-1, 65536):
            with (
                self.subTest(port=port),
                mock.patch.object(gallery, "start_server") as start_server,
                mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    gallery.main(["--port", str(port), "--no-browser"])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--port must be between 0 and 65535", stderr.getvalue())
                start_server.assert_not_called()

    def test_passes_default_and_boundary_ports_to_server(self) -> None:
        for args, port in (([], 8080), (["--port", "0"], 0), (["--port", "65535"], 65535)):
            with self.subTest(port=port), mock.patch.object(gallery, "start_server") as start_server:
                self.assertEqual(gallery.main([*args, "--no-browser"]), 0)
                start_server.assert_called_once_with(
                    port=port, models_dir=None, textures_dir=None, open_browser=False,
                    retry_next_port=not args,
                )

    def test_reports_unavailable_listen_ports(self) -> None:
        cases = ((["--port", "65535"], [65535]), (["--port", "8080"], [8080]), ([], [8080, 8081]))
        for args, ports in cases:
            with (
                self.subTest(ports=ports),
                tempfile.TemporaryDirectory() as assets,
                mock.patch.object(gallery.http.server.HTTPServer, "__init__", side_effect=OSError("address in use")) as bind,
                mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    gallery.main([*args, "--models-dir", assets, "--textures-dir", assets, "--no-browser"])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn(f"can't listen on port {ports[-1]}", stderr.getvalue())
                self.assertEqual([attempt.args[0] for attempt in bind.call_args_list],
                                 [("127.0.0.1", port) for port in ports])

    def test_implicit_port_can_use_the_next_port(self) -> None:
        with (
            tempfile.TemporaryDirectory() as assets,
            mock.patch.object(gallery.http.server.HTTPServer, "__init__",
                              side_effect=[OSError("address in use"), None]) as bind,
            mock.patch.object(gallery.http.server.HTTPServer, "server_port", 8081, create=True),
            mock.patch.object(gallery.http.server.HTTPServer, "serve_forever") as serve,
            mock.patch.object(sys, "stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(gallery.main(["--models-dir", assets, "--textures-dir", assets, "--no-browser"]), 0)
            self.assertEqual([attempt.args[0] for attempt in bind.call_args_list],
                             [("127.0.0.1", 8080), ("127.0.0.1", 8081)])
            self.assertIn("http://localhost:8081", stdout.getvalue())
            serve.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
