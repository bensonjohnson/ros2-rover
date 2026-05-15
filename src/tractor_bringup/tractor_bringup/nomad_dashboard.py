"""Rover-hosted dashboard for the NoMaD runner.

Serves a single page with a live MJPEG stream of the RealSense RGB feed,
with the policy's predicted 8-step waypoint trajectory projected onto the
ground plane and drawn over the image (the same live-path view shown in
the NoMaD project demo).

Threaded http.server only — no Flask/Tornado dependency, matching the
existing RLPD dashboard pattern in rlpd_remote_runner.py.

The node owns a `DashboardState`; it pushes the latest annotated JPEG into
it on every inference tick. The HTTP handler just drains that buffer.
"""

from __future__ import annotations

import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NoMaD Rover</title>
  <style>
    body { background:#111; color:#ddd; font-family:system-ui,sans-serif;
           margin:0; display:flex; flex-direction:column; align-items:center; }
    h1 { font-size:16px; font-weight:600; margin:12px 0 4px; letter-spacing:.04em; }
    .sub { font-size:12px; color:#888; margin-bottom:10px; }
    img { max-width:96vw; border:1px solid #333; image-rendering:auto; }
    .legend { font-size:12px; color:#aaa; margin:10px 0 20px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%;
           vertical-align:middle; margin:0 5px 0 14px; }
  </style>
</head>
<body>
  <h1>NoMaD ROVER &mdash; LIVE PATH</h1>
  <div class="sub">predicted trajectory projected onto the ground plane</div>
  <img src="/stream" alt="live stream">
  <div class="legend">
    <span class="dot" style="background:#39ff14"></span>predicted waypoints
    <span class="dot" style="background:#ff3b3b"></span>lookahead target
  </div>
</body>
</html>
"""


class DashboardState:
    """Thread-safe holder for the most recent annotated JPEG frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stamp = 0.0

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg_bytes
            self._stamp = time.monotonic()

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._jpeg


def _make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/stream'):
                self._serve_stream()
            else:
                self._serve_page()

        def _serve_page(self):
            body = _PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_stream(self):
            self.send_response(200)
            self.send_header(
                'Content-Type', 'multipart/x-mixed-replace; boundary=frame'
            )
            self.send_header('Cache-Control', 'no-cache, private')
            self.end_headers()
            try:
                while True:
                    jpeg = state.get_frame()
                    if jpeg is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(
                            f'Content-Length: {len(jpeg)}\r\n\r\n'.encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                    # ~15 fps stream cap — independent of the policy tick rate.
                    time.sleep(1.0 / 15.0)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return Handler


def start_dashboard_server(state: DashboardState, port: int = 8081) -> ThreadingHTTPServer:
    """Start the dashboard HTTP server.

    Uses ThreadingHTTPServer so the blocking MJPEG /stream handler doesn't
    starve other requests (page reloads, extra tabs). On a fast stop/restart
    the previous process may still be releasing the port, so bind is retried
    for a few seconds before giving up.
    """
    handler = _make_handler(state)
    last_err: OSError | None = None
    for _ in range(12):
        try:
            server = ThreadingHTTPServer(('0.0.0.0', port), handler)
            break
        except OSError as exc:  # port still held by the previous run
            last_err = exc
            time.sleep(0.5)
    else:
        raise last_err if last_err else OSError(f'could not bind port {port}')
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
