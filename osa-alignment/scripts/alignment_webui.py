"""Read-only local dashboard backed by the controller's JSONL trajectory."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image, ImageDraw

from evaluate_alignment import boundary_points, read_image


class Dashboard:
    def __init__(self, log_path, port=8765):
        self.log_path = Path(log_path)
        self.snapshot_path = self.log_path.with_suffix(".html")
        self.template = Path(__file__).with_name("alignment_webui.html").read_text()
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body, content_type = dashboard.html().encode(), "text/html; charset=utf-8"
                elif self.path == "/api/status":
                    body = json.dumps(dashboard.state(), allow_nan=False).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def state(self):
        state = {"settings": {}, "observations": [], "result": None,
                 "message": "Waiting for the first acquisition", "updated": None,
                 "image": None, "image_note": "Waiting for an image"}
        with self.log_path.open() as stream:
            for line in stream:
                # A concurrent append may leave the last record incomplete.
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                state["updated"] = event["time"]
                if event["event"] == "start":
                    state.update(settings=event["settings"],
                                 replay=event["replay"])
                elif event["event"] == "observation":
                    state["observations"].append(event["observation"])
                elif event["event"] == "complete":
                    state["result"] = event["result"]
                if "text" in event:
                    state["message"] = event["text"]
        if state["observations"]:
            try:
                pixels = read_image(state["observations"][-1]["image"])
                low, high = float(pixels.min()), float(pixels.max())
                scaled = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)
                preview = Image.fromarray(np.uint8(np.clip(scaled * 255, 0, 255))).convert("RGB")
                preview.thumbnail((1000, 1000))
                metrics = state["observations"][-1]["metrics"]
                fit = metrics.get("fit")
                overlay_note = "No fitted ellipse available"
                if fit and all(key in fit for key in (
                    "center_x", "center_y", "angle_degrees", "inner_axes", "outer_axes"
                )):
                    draw = ImageDraw.Draw(preview)
                    angles = np.linspace(0, 2 * np.pi, 721)
                    sx, sy = preview.width / pixels.shape[1], preview.height / pixels.shape[0]
                    for boundary, color in (("inner", "#00ffff"), ("outer", "#ff9900")):
                        x, y = boundary_points(fit, boundary, angles)
                        points = list(zip((x + .5) * sx - .5, (y + .5) * sy - .5))
                        draw.line(points, fill=color, width=2)
                    overlay_note = ("Ellipse overlay: inner cyan, outer orange; "
                                    + ("valid fit" if metrics.get("valid") else "fit failed validation"))
                buffer = BytesIO()
                preview.save(buffer, format="PNG")
                state["image"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                state["image_note"] = (f"Display intensity range: {low:.5g}–{high:.5g} (linear). "
                                       + overlay_note)
            except (OSError, ValueError) as exc:
                state["image_note"] = f"Image preview unavailable: {exc}"
        return state

    def html(self, state=None):
        # Escape '<' so image paths and log text cannot terminate the script element.
        payload = json.dumps(state, allow_nan=False).replace("<", "\\u003c")
        return self.template.replace("/* INITIAL_STATE */ null", payload)

    def start(self):
        self.thread.start()

    def close(self):
        try:
            with self.snapshot_path.open("x") as stream:
                stream.write(self.html(self.state()))
            print(f"OSA dashboard snapshot: {self.snapshot_path.resolve()}", flush=True)
        except (OSError, ValueError) as exc:
            print(f"Could not save dashboard snapshot: {exc}", flush=True)
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
