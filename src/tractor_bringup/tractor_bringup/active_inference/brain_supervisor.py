#!/usr/bin/env python3
"""Brain supervisor — boots with the rover, drives awake/sleep from the dashboard.

Runs as a systemd service. Owns the public dashboard port and sits idle until
a mode button is clicked on the page:

    AWAKE  -> spawns `ros2 launch tractor_bringup pc_active_inference.launch.py`
    SLEEP  -> spawns `ros2 run tractor_bringup sleep_consolidator --exit_when_done`
    STOP   -> SIGINTs the child process group (the runner saves the brain on
              SIGINT), escalating to SIGTERM/SIGKILL if it lingers
    UPDATE -> `git pull --ff-only` + `colcon build` in the workspace; if code
              actually changed and built, the supervisor re-execs itself so
              everything (supervisor included) runs the new code. The same
              update runs automatically on startup.

Lidar USB power (--lidar-power-control, OFF by default): the STL19P spins
whenever it has 5V (broadcast-only protocol, no motor-stop command), so the
supervisor can cut the lidar's hub port with `sudo uhubctl` while idle and
power it back up (waiting for /dev/ttyUSB* to enumerate) before awake mode.
Requires a sudoers entry (`ubuntu ALL=(root) NOPASSWD: /usr/sbin/uhubctl`)
AND a hub with real per-port VBUS switching — the rover's onboard Genesys
hub only gates the data lines (device disconnects but the motor keeps
spinning), so this stays disabled until the lidar is behind a hub with
actual power MOSFETs (see the uhubctl compatible-hubs list).

The child processes host their own PCDashboardState server on an internal
port; the supervisor proxies /state to it and injects `supervisor_mode`, so
the one page at the public port works in every state. While idle, the last
state seen from the previous run is kept on screen.

No rclpy dependency — children are plain subprocesses in their own session
(process group), exactly like running the launch file by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from tractor_bringup.active_inference.pc_dashboard import _PAGE


class BrainSupervisor:
    BUILD_PACKAGES = ["tractor_bringup", "tractor_control", "tractor_sensors"]

    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._mode = "idle"            # idle | awake | sleep | updating
        self._child: subprocess.Popen | None = None
        self._child_started = 0.0
        self._last_state: dict = {}    # last good /state from a child
        self._last_state_t = 0.0
        self._note = ""                # short status line for the page
        os.makedirs(self.args.log_dir, exist_ok=True)
        threading.Thread(target=self._reaper_loop, daemon=True).start()
        # Boot state: nothing runs yet, so the lidar shouldn't spin.
        self._lidar_power(False)

    # ---- lidar USB power -----------------------------------------------------

    def _lidar_power(self, on: bool) -> bool:
        """Switch the lidar's USB hub port via uhubctl (best effort)."""
        if not self.args.lidar_power_control:
            return True
        ok = True
        for hub in self.args.lidar_usb_hubs.split(","):
            hub = hub.strip()
            if not hub:
                continue
            cmd = ["sudo", "-n", "uhubctl", "-l", hub,
                   "-p", str(self.args.lidar_usb_port),
                   "-a", "on" if on else "off", "-f"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=15)
                if r.returncode != 0:
                    ok = False
                    print(f"[supervisor] uhubctl {hub} failed: "
                          f"{(r.stderr or r.stdout).strip()}", flush=True)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[supervisor] uhubctl error: {e}", flush=True)
        print(f"[supervisor] lidar USB power {'ON' if on else 'OFF'}"
              f"{'' if ok else ' (with errors)'}", flush=True)
        return ok

    def _wait_for_lidar(self, timeout: float = 12.0) -> bool:
        """Wait for the lidar serial device to enumerate after power-up."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(self.args.lidar_port):
                time.sleep(1.0)   # let the tty settle before the driver opens it
                return True
            time.sleep(0.3)
        return False

    # ---- child lifecycle ---------------------------------------------------

    def start(self, mode: str) -> tuple[bool, str]:
        with self._lock:
            if self._mode == "updating":
                return False, "update in progress"
            if self._child is not None and self._child.poll() is None:
                return False, f"already running: {self._mode}"
            if mode == "awake":
                self._note = "powering up lidar..."

        # Awake needs the lidar; power its port up and wait for the tty to
        # enumerate (done outside the lock — it takes a few seconds and the
        # page keeps polling meanwhile).
        if mode == "awake" and self.args.lidar_power_control:
            self._lidar_power(True)
            if not self._wait_for_lidar():
                self._lidar_power(False)
                with self._lock:
                    self._note = f"lidar did not enumerate at {self.args.lidar_port}"
                return False, self._note

        with self._lock:
            if self._mode == "updating":
                return False, "update in progress"
            if self._child is not None and self._child.poll() is None:
                return False, f"already running: {self._mode}"
            if mode == "awake":
                cmd = [
                    "ros2", "launch", "tractor_bringup",
                    "pc_active_inference.launch.py",
                    f"action_scale:={self.args.action_scale}",
                    f"control_rate_hz:={self.args.control_rate}",
                    f"lidar_port:={self.args.lidar_port}",
                    f"dashboard_port:={self.args.child_port}",
                ]
            elif mode == "sleep":
                cmd = [
                    "ros2", "run", "tractor_bringup", "sleep_consolidator",
                    "--model_path", self.args.model_path,
                    "--experience_log_path", self.args.experience_log_path,
                    "--dashboard_port", str(self.args.child_port),
                    "--exit_when_done",
                ]
            else:
                return False, f"unknown mode: {mode}"

            log_path = os.path.join(
                self.args.log_dir,
                f"pc_brain_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log")
            log_f = open(log_path, "ab")
            # New session = own process group, so stop() can signal the whole
            # ros2-launch tree at once.
            self._child = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                start_new_session=True)
            self._child_started = time.monotonic()
            self._mode = mode
            self._note = f"{mode} started, log: {log_path}"
            print(f"[supervisor] {self._note}", flush=True)
            return True, self._note

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if self._mode == "updating":
                return False, "update in progress — wait for it to finish"
            child = self._child
            if child is None or child.poll() is not None:
                self._mode = "idle"
                return True, "nothing running"
            pgid = os.getpgid(child.pid)
            mode = self._mode
        # Signal outside the lock: the awake runner saves the brain on SIGINT,
        # which can take a moment.
        print(f"[supervisor] stopping {mode} (pgid {pgid})", flush=True)
        for sig, grace in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    break
                time.sleep(0.2)
            if child.poll() is not None:
                break
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5.0)
        with self._lock:
            self._child = None
            self._mode = "idle"
            self._note = f"{mode} stopped"
        print(f"[supervisor] {mode} stopped", flush=True)
        self._lidar_power(False)   # idle again: stop the lidar motor
        return True, f"{mode} stopped"

    # ---- code update ---------------------------------------------------------

    def update(self) -> tuple[bool, str]:
        with self._lock:
            if self._mode == "updating":
                return False, "update already in progress"
            if self._child is not None and self._child.poll() is None:
                return False, f"stop {self._mode} before updating"
            ws = self.args.workspace
            if not os.path.isdir(os.path.join(ws, ".git")):
                return False, f"not a git repo: {ws}"
            self._mode = "updating"
            self._note = "updating: git pull..."
        threading.Thread(target=self._do_update, daemon=True).start()
        return True, "update started"

    def _do_update(self):
        ws = self.args.workspace
        log_path = os.path.join(
            self.args.log_dir, f"update_{time.strftime('%Y%m%d_%H%M%S')}.log")

        def run(cmd, timeout):
            with open(log_path, "ab") as lf:
                lf.write(f"\n$ {' '.join(cmd)}\n".encode())
                lf.flush()
                return subprocess.run(
                    cmd, cwd=ws, stdout=lf, stderr=subprocess.STDOUT,
                    timeout=timeout).returncode

        try:
            head_before = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ws,
                capture_output=True, text=True, timeout=30).stdout.strip()
            if run(["git", "pull", "--ff-only"], timeout=120) != 0:
                self._finish_update(f"git pull failed — see {log_path}")
                return
            head_after = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ws,
                capture_output=True, text=True, timeout=30).stdout.strip()

            if head_after == head_before:
                self._finish_update("code already up to date")
                return

            with self._lock:
                self._note = f"updating: building {head_after[:8]}..."
            rc = run(["colcon", "build",
                      "--packages-select", *self.BUILD_PACKAGES,
                      "--cmake-args", "-DCMAKE_BUILD_TYPE=Release"],
                     timeout=900)
            if rc != 0:
                self._finish_update(
                    f"build FAILED at {head_after[:8]} — see {log_path}; "
                    f"still running old code")
                return

            print(f"[supervisor] updated to {head_after[:8]}, restarting "
                  f"to load new code", flush=True)
            # Replace this process so the supervisor itself runs the new
            # code. PID is preserved (systemd-friendly); the page's poll
            # loop reconnects on its own. --no-startup-update prevents a
            # redundant second pull after the re-exec.
            argv = [a for a in sys.argv if a != "--no-startup-update"]
            argv.append("--no-startup-update")
            os.execv(sys.executable, [sys.executable] + argv)
        except Exception as e:  # noqa: BLE001
            self._finish_update(f"update error: {e}")

    def _finish_update(self, note: str):
        with self._lock:
            self._mode = "idle"
            self._note = note
        print(f"[supervisor] {note}", flush=True)

    def _reaper_loop(self):
        """Notice children that exit on their own (sleep finishing, crashes)."""
        while True:
            time.sleep(1.0)
            with self._lock:
                child, mode = self._child, self._mode
            if child is None or mode == "idle":
                continue
            rc = child.poll()
            if rc is None:
                continue
            with self._lock:
                self._child = None
                prev = self._mode
                self._mode = "idle"
                if prev == "sleep" and rc == 0:
                    self._note = "sleep cycle completed"
                else:
                    self._note = f"{prev} exited (code {rc})"
            print(f"[supervisor] {self._note}", flush=True)
            self._lidar_power(False)   # idle again: stop the lidar motor

    # ---- state for the page --------------------------------------------------

    def state_json(self) -> bytes:
        """Proxy the child's /state, or serve the cached last state while idle."""
        with self._lock:
            mode = self._mode
        if mode != "idle":
            try:
                url = f"http://127.0.0.1:{self.args.child_port}/state"
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    state = json.loads(resp.read())
                with self._lock:
                    self._last_state = state
                    self._last_state_t = time.monotonic()
            except Exception:
                # Child still booting (lidar/imu bringup takes a few seconds)
                # or busy — fall through to the cache.
                pass
        with self._lock:
            state = dict(self._last_state)
            if self._last_state_t:
                state["age"] = time.monotonic() - self._last_state_t
            state["supervisor_mode"] = self._mode
            state["supervisor_note"] = self._note
            if self._mode in ("idle", "updating"):
                state["mode"] = self._mode
        return json.dumps(state).encode()

    def control(self, action: str) -> tuple[bool, str]:
        if action in ("awake", "sleep"):
            return self.start(action)
        if action == "stop":
            return self.stop()
        if action == "update":
            return self.update()
        return False, f"unknown action: {action}"


def _make_handler(sup: BrainSupervisor):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(200, sup.state_json(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, _PAGE.encode(), "text/html")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/control"):
                self._send(404, b"", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                action = str(payload.get("action", ""))
            except Exception:
                self._send(400, b'{"ok": false, "msg": "bad request"}',
                           "application/json")
                return
            ok, msg = sup.control(action)
            body = json.dumps({"ok": ok, "msg": msg}).encode()
            self._send(200 if ok else 409, body, "application/json")

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="PC rover brain supervisor")
    parser.add_argument("--port", type=int, default=8082,
                        help="Public dashboard/control port")
    parser.add_argument("--child-port", type=int, default=8083,
                        help="Internal dashboard port the brain/consolidator binds")
    parser.add_argument("--action-scale", dest="action_scale", default="0.6")
    parser.add_argument("--control-rate", dest="control_rate", default="15.0")
    parser.add_argument("--lidar-port", dest="lidar_port", default="/dev/ttyUSB0")
    parser.add_argument("--model-path", dest="model_path",
                        default=os.path.expanduser("~/.ros/pnn_brain.pt"))
    parser.add_argument("--experience-log-path", dest="experience_log_path",
                        default=os.path.expanduser("~/.ros/pnn_experience.jsonl"))
    parser.add_argument("--log-dir", dest="log_dir",
                        default=os.path.expanduser("~/.ros/pc_brain_logs"))
    parser.add_argument("--workspace", default=os.getcwd(),
                        help="ros2-rover workspace root (git repo, colcon ws)")
    parser.add_argument("--lidar-usb-hubs", dest="lidar_usb_hubs",
                        default="1-1,2-1",
                        help="uhubctl hub locations for the lidar port "
                             "(USB2 hub and its USB3 sibling)")
    parser.add_argument("--lidar-usb-port", dest="lidar_usb_port",
                        type=int, default=4,
                        help="Hub port number the lidar adapter is plugged into")
    parser.add_argument("--lidar-power-control", dest="lidar_power_control",
                        action="store_true", default=False,
                        help="Switch the lidar's USB port power via uhubctl when "
                             "idle (needs a hub with REAL per-port VBUS switching; "
                             "the onboard hub only gates data, motor keeps spinning)")
    parser.add_argument("--no-startup-update", action="store_true",
                        help="Skip the automatic git pull + build on startup")
    args = parser.parse_args(argv)

    sup = BrainSupervisor(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _make_handler(sup))
    server.daemon_threads = True
    print(f"[supervisor] dashboard+control on http://0.0.0.0:{args.port} "
          f"(child port {args.child_port}); idle — waiting for a mode button",
          flush=True)

    if not args.no_startup_update:
        sup.update()

    def shutdown(signum, frame):
        print("[supervisor] shutting down", flush=True)
        sup.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    server.serve_forever()


if __name__ == "__main__":
    main()
