#!/usr/bin/env python3
"""Rover control service — one page to start, stop, and hard-stop the rover.

Boot-time systemd service on :8090. Composes the proven BrainSupervisor
(child lifecycle, git-pull+colcon update, /state proxy) and adds what the
field actually needs from a single button press:

  HARDSTOP  -> spawns hardstop_hold (own process group, 20 Hz latch of
               /emergency_stop=true + zeroed track_cmd/cmd_vel) AND stops the
               brain. The latch is a FILE (~/.ros/pc_brain_hardstop.latch):
               if this service crashes or the hold process is killed, whoever
               comes back up re-engages it. RELEASE deletes the file. Safety
               state survives process death; only an explicit release clears.
  AWAKE     -> ros2 launch tractor_bringup pc_active_inference.launch.py
               (the brain's rich dashboard stays on the child port, :8082)
  SLEEP     -> sleep_consolidator --exit_when_done
  STOP      -> SIGINT the child group (runner saves the brain)
  UPDATE    -> git pull + colcon build (inherited; DNS-race retries)

Health cards come from an in-process rclpy subscriber thread (best-effort QoS
so it matches whatever the producers use): /scan hz (the serial-theft
detector — must read 10.0), /imu/data, battery, /safety_monitor_status,
/emergency_stop. No rclpy? The page still works, cards show 'n/a'.

Logs: last lines of the newest ~/.ros/pc_brain_logs/* file, viewable in the
page — no ssh to find out why the brain died.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Same package as brain_supervisor; no rclpy at import time (the health
# thread imports it lazily so the control plane survives a broken ROS env).
from tractor_bringup.active_inference.brain_supervisor import BrainSupervisor

LATCH_FILE_DEFAULT = os.path.expanduser("~/.ros/pc_brain_hardstop.latch")


# --------------------------------------------------------------------------
# Topic health monitor (optional rclpy thread)
# --------------------------------------------------------------------------

class HealthMonitor:
    """Best-effort subscriptions that record message rate/age per topic.

    One entry per watched topic: {'last': monotonic ts or None,
    'ts': deque of arrival times, 'val': last data}.
    """

    def __init__(self):
        self.available = False
        self.error = ""
        self._lock = threading.Lock()
        self._watchdog_stop = False
        self._spin_thread: threading.Thread | None = None
        self._born = time.monotonic()
        self._t: dict = {
            k: {"last": None, "ts": deque(maxlen=64), "val": None}
            for k in ("scan", "imu", "battery", "safety", "estop", "track")}

    def _hit(self, key, val=None):
        with self._lock:
            e = self._t[key]
            e["last"] = time.monotonic()
            e["ts"].append(e["last"])
            if val is not None:
                e["val"] = val

    def _any_publisher_alive(self) -> bool:
        """True if the DDS graph shows a publisher on any watched topic.
        Distinguishes 'nothing is running' (idle rover — normal, don't
        rebuild) from 'publisher up but our subscription is deaf' (wedge)."""
        node = getattr(self, "_node", None)
        if node is None:
            return False
        topics = {"scan": "/scan", "imu": "/imu/data",
                  "battery": "/battery_percentage",
                  "safety": "/safety_monitor_status",
                  "estop": "/emergency_stop", "track": "/track_cmd"}
        try:
            names = {n for n, _types in node.get_topic_names_and_types()}
            for t in topics.values():
                if t in names and node.count_publishers(t) > 0:
                    return True
        except Exception:  # noqa: BLE001  (graph query raced a teardown)
            return False
        return False

    def snapshot(self) -> dict:
        now = time.monotonic()
        out = {"rclpy": self.available}
        if not self.available and self.error:
            out["rclpy_error"] = self.error
        with self._lock:
            for key, e in self._t.items():
                d = {}
                if e["last"] is not None:
                    d["age"] = round(now - e["last"], 2)
                    window = [t for t in e["ts"] if now - t <= 5.0]
                    if len(window) >= 2:
                        span = window[-1] - window[0]
                        d["hz"] = round((len(window) - 1) / span, 2) if span > 0 else None
                    d["val"] = e["val"]
                out[key] = d or {}
        return out

    def _run(self):
        try:
            import rclpy
            from rclpy.executors import ExternalShutdownException
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import LaserScan
            from sensor_msgs.msg import Imu
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Bool, Float32, Float32MultiArray, String
        except Exception as e:  # noqa: BLE001
            self.error = f"rclpy unavailable: {e}"
            # Maybe the ROS env wasn't ready at boot; _run_supervised will
            # restart us — just wait out this cycle.
            time.sleep(30.0)
            return

        class _Watch(Node):
            pass

        # NOTE: rclpy.init() may already be up from a previous cycle; guard it.
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass
        node = _Watch("rover_control_health")
        Q = qos_profile_sensor_data  # best effort: matches reliable pubs too
        node.create_subscription(LaserScan, "/scan",
                                 lambda m: self._hit("scan"), Q)
        node.create_subscription(Imu, "/imu/data",
                                 lambda m: self._hit("imu"), Q)
        node.create_subscription(Float32, "/battery_percentage",
                                 lambda m: self._hit("battery", round(m.data, 1)), Q)
        node.create_subscription(String, "/safety_monitor_status",
                                 lambda m: self._hit("safety", m.data[:120]), Q)
        node.create_subscription(Bool, "/emergency_stop",
                                 lambda m: self._hit("estop", bool(m.data)), Q)
        node.create_subscription(Float32MultiArray, "/track_cmd",
                                 lambda m: self._hit("track", [round(x, 2) for x in m.data[:2]]), Q)
        self.available = True
        self._node = node        # exposed for the watchdog's graph queries
        try:
            rclpy.spin(node)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass
        finally:
            self._node = None
            try:
                node.destroy_node()
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self.available = False

    def _run_supervised(self):
        """Keep _run() alive. A graph-state bug (seen on real hardware: the
        spin wedging when the node subscribed before any publisher existed at
        boot, leaving every card empty forever) self-heals in <=120 s: if no
        watched topic has delivered anything for two minutes, invalidate the
        ROS context (unblocks spin()), tear down, and rebuild the
        subscriptions from scratch.

        NOTE: 'no traffic' is NORMAL while the rover is idle (lidar and motor
        driver are down between modes), so an un-stuck rebuild is not a
        failure — giving up would kill the cards when AWAKE later brings
        traffic back. We only give up on a truly unbreakable spin (the thread
        cannot be released despite rclpy.shutdown(), 5x in a row)."""
        unbreakable = 0
        while not self._watchdog_stop:
            self._spin_thread = threading.Thread(target=self._run, daemon=True)
            self._spin_thread.start()
            t_start = time.monotonic()
            wedged = False
            while self._spin_thread.is_alive() and not self._watchdog_stop:
                time.sleep(5.0)
                with self._lock:
                    fresh = any(e["last"] and time.monotonic() - e["last"] < 120.0
                                for e in self._t.values())
                if fresh or time.monotonic() - t_start <= 120.0:
                    continue
                # No traffic for 2 minutes. That is NORMAL while the rover is
                # idle (lidar/motor drivers are down between modes) — only
                # rebuild when the graph says someone IS publishing and we
                # still hear nothing (a real wedge).
                if not self._any_publisher_alive():
                    continue
                wedged = True
                print("[rover-control] health watchdog: publishers up but no "
                      "traffic for 120 s — rebuilding ROS subscriptions",
                      flush=True)
                break
            if self._watchdog_stop:
                break
            if wedged:
                try:  # invalidate the context so spin() returns
                    import rclpy
                    rclpy.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            self._spin_thread.join(timeout=15.0)
            if self._spin_thread.is_alive():
                unbreakable += 1
                if unbreakable >= 5:
                    print("[rover-control] health: spin unbreakable after 5 "
                          "attempts (cards stay '-')", flush=True)
                    return
                time.sleep(30.0)     # couldn't unstick; back off before retry
            else:
                unbreakable = 0      # clean teardown: idle or healed
                if not self.available:
                    time.sleep(5.0)  # crashed fast; don't spin CPU

    def start(self):
        self._born = time.monotonic()
        threading.Thread(target=self._run_supervised, daemon=True).start()

    def shutdown(self):
        self._watchdog_stop = True


# --------------------------------------------------------------------------
# HARDSTOP manager — latch survives process death via a marker file
# --------------------------------------------------------------------------

class Hardstop:
    def __init__(self, latch_file: str, ros_prefix: str):
        self.latch_file = latch_file
        self.ros_prefix = ros_prefix          # shell prefix sourcing ROS+install
        self._lock = threading.Lock()
        self._child: subprocess.Popen | None = None
        # Re-engage if a previous session left the latch set (crash safety).
        if os.path.exists(latch_file):
            print("[rover-control] hardstop latch file present — re-engaging",
                  flush=True)
            self.engage(stop_brain=None)

        def _guard():
            # Hold process must never stay dead while the latch is set.
            while True:
                time.sleep(1.0)
                with self._lock:
                    latched = os.path.exists(latch_file)
                    dead = self._child is None or self._child.poll() is not None
                if latched and dead:
                    print("[rover-control] hold process dead while latched — "
                          "respawning", flush=True)
                    self._spawn()
        threading.Thread(target=_guard, daemon=True).start()

    @property
    def engaged(self) -> bool:
        return os.path.exists(self.latch_file)

    def _spawn(self):
        cmd = (f"{self.ros_prefix} exec ros2 run tractor_bringup hardstop_hold")
        log_f = open(os.devnull, "wb")
        with self._lock:
            if self._child is not None and self._child.poll() is None:
                return
            self._child = subprocess.Popen(
                ["/bin/bash", "-c", cmd], stdout=log_f,
                stderr=subprocess.STDOUT, start_new_session=True)

    def engage(self, stop_brain):
        open(self.latch_file, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        if stop_brain is not None:
            try:
                stop_brain()
            except Exception as e:  # noqa: BLE001
                print(f"[rover-control] stop during hardstop failed: {e}", flush=True)
        self._spawn()
        print("[rover-control] HARDSTOP ENGAGED", flush=True)
        return True, "HARDSTOP engaged — rover held motionless"

    def release(self, stop_brain=None) -> tuple[bool, str]:
        if not self.engaged:
            return False, "not engaged"
        with self._lock:
            ch = self._child
            self._child = None
        if ch is not None and ch.poll() is None:
            try:
                os.killpg(os.getpgid(ch.pid), signal.SIGINT)
                ch.wait(timeout=5.0)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(ch.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        try:
            os.remove(self.latch_file)
        except FileNotFoundError:
            pass
        # Deliberate-release policy: releasing also stops the brain, so the
        # rover is motionless after RELEASE. Motion resumes only on the next
        # explicit AWAKE — releasing must never leave the rover driving.
        if stop_brain is not None:
            try:
                stop_brain()
            except Exception as e:  # noqa: BLE001
                print(f"[rover-control] stop during release failed: {e}", flush=True)
        print("[rover-control] HARDSTOP released", flush=True)
        return True, "HARDSTOP released — rover stopped; press AWAKE to resume"


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rover Control</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#0b0d10;color:#dbe2ec;font:14px system-ui,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:14px}
  h1{font-size:16px;letter-spacing:.06em;margin:4px 0 12px;color:#9aa4b2}
  .bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
  button{font:inherit;font-weight:600;padding:12px 18px;border-radius:10px;border:1px solid #2a3038;
         background:#161a20;color:#dbe2ec;cursor:pointer}
  button:disabled{opacity:.35;cursor:default}
  #btn_hard{background:#5c1414;border-color:#8b1e1e;color:#ffdad8;font-size:16px;padding:14px 26px}
  #btn_hard.armed{background:#8b1e1e;box-shadow:0 0 14px #b32222 inset}
  #btn_release{border-color:#3a4654}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:12px}
  .card{background:#14171c;border:1px solid #232830;border-radius:8px;padding:8px 10px}
  .lab{font-size:10px;letter-spacing:.07em;color:#6b7480}
  .val{font-size:19px;font-variant-numeric:tabular-nums;margin-top:2px}
  .ok{color:#39ff14}.warn{color:#ffd043}.dead{color:#ff5b5b}.dim{color:#6b7480}
  #mode{font-weight:700}
  .note{margin:2px 0 12px;font-size:12px;color:#ff9d3b;min-height:16px}
  .sec{font-size:11px;letter-spacing:.08em;color:#6b7480;margin:14px 0 6px}
  select,pre{width:100%;box-sizing:border-box}
  select{background:#14171c;color:#dbe2ec;border:1px solid #232830;border-radius:6px;padding:6px}
  pre{background:#0f1216;border:1px solid #232830;border-radius:8px;padding:10px;
      font:11px/1.45 ui-monospace,monospace;overflow:auto;max-height:320px;white-space:pre-wrap}
  a{color:#5bc0ff}
  #estop_banner{display:none;background:#5c1414;border:1px solid #8b1e1e;color:#ffdad8;
                padding:10px;border-radius:10px;margin-bottom:12px;font-weight:700;text-align:center}
</style></head><body><div class="wrap">
<h1>ROVER CONTROL</h1>
<div id="estop_banner">HARDSTOP ENGAGED — rover is held motionless</div>
<div class="bar">
  <button id="btn_awake">AWAKE</button>
  <button id="btn_sleep">SLEEP</button>
  <button id="btn_stop">STOP</button>
  <button id="btn_hard">HARDSTOP</button>
  <button id="btn_release" disabled>RELEASE</button>
  <button id="btn_update">UPDATE</button>
</div>
<div class="note" id="note">&nbsp;</div>
<div class="grid">
  <div class="card"><div class="lab">MODE</div><div class="val" id="c_mode">-</div></div>
  <div class="card"><div class="lab">SCAN Hz</div><div class="val" id="c_scan">-</div></div>
  <div class="card"><div class="lab">IMU</div><div class="val" id="c_imu">-</div></div>
  <div class="card"><div class="lab">BATTERY</div><div class="val" id="c_batt">-</div></div>
  <div class="card"><div class="lab">SAFETY MON</div><div class="val" id="c_safe">-</div></div>
  <div class="card"><div class="lab">ESTOP</div><div class="val" id="c_estop">-</div></div>
  <div class="card"><div class="lab">BRAIN</div><div class="val" id="c_brain">-</div></div>
</div>
<div class="sec">BRAIN (when awake) &nbsp;·&nbsp; <a id="brain_link" href="#">open brain dashboard →</a></div>
<div class="grid">
  <div class="card"><div class="lab">STEP</div><div class="val dim" id="b_step">-</div></div>
  <div class="card"><div class="lab">FREE ENERGY</div><div class="val dim" id="b_F">-</div></div>
  <div class="card"><div class="lab">NOVELTY</div><div class="val dim" id="b_nov">-</div></div>
  <div class="card"><div class="lab">PLACES</div><div class="val dim" id="b_pl">-</div></div>
</div>
<div class="sec">LOGS</div>
<select id="logsel"><option value="">(no logs)</option></select>
<pre id="logbox">select a log…</pre>
</div>
<script>
const $=id=>document.getElementById(id);
const ctl=(action)=>fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({action})}).then(r=>r.json()).then(j=>{$('note').textContent=j.msg});
$('btn_awake').onclick=()=>{if(confirm('Start AWAKE? The rover drives autonomously.'))ctl('awake')};
$('btn_sleep').onclick=()=>ctl('sleep');
$('btn_stop').onclick=()=>{if(confirm('Stop the current mode? (SIGINT saves the brain)'))ctl('stop')};
$('btn_hard').onclick=()=>ctl('hardstop');          // NEVER confirm() a stop
$('btn_release').onclick=()=>{if(confirm('Release HARDSTOP? Motion commands go live again.'))ctl('release')};
$('btn_update').onclick=()=>{if(confirm('Pull latest code and rebuild?'))ctl('update')};

function put(id,text,cls){const e=$(id);e.textContent=text;e.className='val '+(cls||'');}
function ago(s,lo,hi){ // cls by age thresholds
  return s==null?'dead':(s<=lo?'ok':(s<=hi?'warn':'dead'));}

async function tick(){
  let s;try{s=await (await fetch('/state')).json();}catch(e){put('c_mode','OFFLINE','dead');return;}
  put('c_mode',s.mode||'?',s.mode==='awake'?'ok':(s.mode==='idle'?'dim':'warn'));
  const h=s.health||{};
  const scan=h.scan||{},imu=h.imu||{},batt=h.battery||{},safe=h.safety||{},est=h.estop||{};
  put('c_scan',scan.hz!=null?scan.hz.toFixed(1):'-',
      scan.hz==null?'dead':(Math.abs(scan.hz-10)<1.5?'ok':(scan.hz>7?'warn':'dead')));
  put('c_imu',imu.age!=null?(imu.age<2?'up':'stale'):'-',ago(imu.age,2,5));
  put('c_batt',batt.val!=null?batt.val+'%':'-',batt.val==null?'dead':(batt.val>20?'ok':'warn'));
  put('c_safe',safe.age!=null?(safe.age<3?'live':'stale'):'-',ago(safe.age,3,8));
  put('c_estop',est.val===true?'HELD':(est.age!=null?'clear':'-'),est.val===true?'dead':'dim');
  put('c_brain',s.mode==='awake'?(s.brain&&s.brain.step!=null?'running':'starting'):'idle',
      s.mode==='awake'?'ok':'dim');
  const b=s.brain||{};
  put('b_step',b.step!=null?b.step:'-','dim');
  put('b_F',b.F!=null?b.F:'-','dim');
  put('b_nov',b.novelty!=null?b.novelty:'-','dim');
  put('b_pl',b.places_n!=null?b.places_n:'-','dim');
  $('estop_banner').style.display=s.hardstop?'block':'none';
  $('btn_hard').classList.toggle('armed',!!s.hardstop);
  $('btn_hard').disabled=!!s.hardstop;
  $('btn_release').disabled=!s.hardstop;
  const idle=(s.mode==='idle'&&!s.hardstop);
  $('btn_awake').disabled=!idle;$('btn_sleep').disabled=!idle;$('btn_update').disabled=!idle;
  $('btn_stop').disabled=(s.mode==='idle'||s.mode==='updating');
  $('brain_link').style.display=(s.mode==='awake')?'inline':'none';
  $('brain_link').href='http://'+location.hostname+':'+(s.child_port||8082);
  if(s.note)$('note').textContent=s.note;
}
async function loadLogs(){
  let l;try{l=await (await fetch('/logs')).json();}catch(e){return;}
  const sel=$('logsel');const cur=sel.value;
  sel.innerHTML='';
  (l.files||[]).forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent=f;sel.appendChild(o);});
  if(cur&&l.files.includes(cur))sel.value=cur;else if(l.files.length)sel.value=l.files[0];
  if(sel.value){
    try{const t=await (await fetch('/logs?file='+encodeURIComponent(sel.value))).json();
        $('logbox').textContent=(t.lines||[]).join('\\n')||'(empty)';}catch(e){}
  } else $('logbox').textContent='(no logs)';
}
$('logsel').onchange=loadLogs;
setInterval(tick,1000);tick();
setInterval(loadLogs,10000);loadLogs();
</script></body></html>
"""


# --------------------------------------------------------------------------
# Service composition
# --------------------------------------------------------------------------

def _tail(path: str, n: int = 200) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read().decode("utf-8", "replace")
        return data.splitlines()[-n:]
    except OSError as e:
        return [f"(cannot read: {e})"]


class Handler(BaseHTTPRequestHandler):
    # Injected via class attrs in main() (sup, hard, health, args).
    def _send(self, code, body: bytes, ctype):
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
        if self.path in ("/", "/index.html"):
            self._send(200, _PAGE.encode(), "text/html")
        elif self.path.startswith("/state"):
            try:
                state = json.loads(self.sup.state_json())
            except Exception:  # noqa: BLE001
                state = {}
            # Slim brain summary for the control page (full page is on child port)
            keep = ("step", "F", "novelty", "novelty_target", "places_n", "mode")
            state["brain"] = {k: state.get(k) for k in keep if k in state}
            state["hardstop"] = self.hard.engaged
            state["health"] = self.health.snapshot()
            state["child_port"] = self.args.child_port
            state["note"] = state.get("supervisor_note", "")
            # Control-page truth: the supervisor's mode (idle/awake/sleep/
            # updating), never the brain-internal mode the child reports.
            state["mode"] = state.get("supervisor_mode", "idle")
            self._send(200, json.dumps(state).encode(), "application/json")
        elif self.path.startswith("/logs"):
            q = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(q)
            if "file" in params:
                name = os.path.basename(params["file"][0])   # jail to log_dir
                self._send(200, json.dumps(
                    {"lines": _tail(os.path.join(self.args.log_dir, name))}
                ).encode(), "application/json")
            else:
                try:
                    names = [f for f in os.listdir(self.args.log_dir)
                             if f.endswith(".log")]
                    names.sort(key=lambda f: os.path.getmtime(
                        os.path.join(self.args.log_dir, f)), reverse=True)
                except OSError:
                    names = []
                self._send(200, json.dumps({"files": names[:40]}).encode(),
                           "application/json")
            return
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
        except Exception:  # noqa: BLE001
            self._send(400, b'{"ok": false, "msg": "bad request"}',
                       "application/json")
            return
        if action == "hardstop":
            ok, msg = self.hard.engage(stop_brain=lambda: self.sup.control("stop"))
        elif action == "release":
            ok, msg = self.hard.release(stop_brain=lambda: self.sup.control("stop"))
        else:
            if self.hard.engaged and action != "stop":
                self._send(409, json.dumps(
                    {"ok": False,
                     "msg": "HARDSTOP engaged — RELEASE first before "
                            "starting modes"}).encode(), "application/json")
                return
            ok, msg = self.sup.control(action)
        body = json.dumps({"ok": ok, "msg": msg}).encode()
        self._send(200 if ok else 409, body, "application/json")

    def log_message(self, *args):
        pass

    # class-injected attributes (documented for readers/Pyright)
    sup: BrainSupervisor
    hard: Hardstop
    health: HealthMonitor
    args: argparse.Namespace


def main(argv=None):
    global_parser = argparse.ArgumentParser(description="Rover control service")
    global_parser.add_argument("--port", type=int, default=8090)
    global_parser.add_argument("--child-port", dest="child_port", type=int,
                               default=8082,
                               help="Brain dashboard port (when awake)")
    global_parser.add_argument("--workspace", default=os.getcwd())
    global_parser.add_argument("--log-dir", dest="log_dir",
                               default=os.path.expanduser("~/.ros/pc_brain_logs"))
    global_parser.add_argument("--latch-file", dest="latch_file",
                               default=LATCH_FILE_DEFAULT)
    global_parser.add_argument("--ros-setup", dest="ros_setup",
                               default="/opt/ros/jazzy/setup.bash")
    # Pass-through to BrainSupervisor
    global_parser.add_argument("--action-scale", dest="action_scale", default="0.6")
    global_parser.add_argument("--control-rate", dest="control_rate", default="15.0")
    global_parser.add_argument("--lidar-port", dest="lidar_port",
                               default="/dev/ttyUSB0")
    global_parser.add_argument("--imu-type", dest="imu_type", default="bno085",
                               choices=["lsm9ds1", "bno085"])
    global_parser.add_argument("--model-path", dest="model_path",
                               default=os.path.expanduser("~/.ros/pnn_brain.pt"))
    global_parser.add_argument("--experience-log-path", dest="experience_log_path",
                               default=os.path.expanduser("~/.ros/pnn_experience.jsonl"))
    global_parser.add_argument("--lidar-usb-hubs", dest="lidar_usb_hubs",
                               default="1-1,2-1")
    global_parser.add_argument("--lidar-usb-port", dest="lidar_usb_port",
                               type=int, default=4)
    global_parser.add_argument("--lidar-power-control", dest="lidar_power_control",
                               action="store_true", default=False)
    global_parser.add_argument("--no-startup-update", action="store_true")
    args = global_parser.parse_args(argv)

    os.makedirs(args.log_dir, exist_ok=True)

    # BrainSupervisor expects its own arg names; hand it a compatible view.
    sup_args = argparse.Namespace(
        port=args.child_port,          # unused by lifecycle code, harmless
        child_port=args.child_port,
        action_scale=args.action_scale,
        control_rate=args.control_rate,
        lidar_port=args.lidar_port,
        imu_type=args.imu_type,
        model_path=args.model_path,
        experience_log_path=args.experience_log_path,
        log_dir=args.log_dir,
        workspace=args.workspace,
        lidar_usb_hubs=args.lidar_usb_hubs,
        lidar_usb_port=args.lidar_usb_port,
        lidar_power_control=args.lidar_power_control,
        no_startup_update=args.no_startup_update,
    )
    sup = BrainSupervisor(sup_args)

    ros_prefix = f"source {args.ros_setup} && source {args.workspace}/install/setup.bash &&"
    hard = Hardstop(latch_file=args.latch_file, ros_prefix=ros_prefix)

    health = HealthMonitor()
    health.start()

    Handler.sup = sup
    Handler.hard = hard
    Handler.health = health
    Handler.args = args

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"[rover-control] control page on http://0.0.0.0:{args.port} "
          f"(brain dashboard at :{args.child_port} when awake); "
          f"hardstop={'ENGAGED' if hard.engaged else 'clear'}", flush=True)

    if not args.no_startup_update:
        sup.update()

    def shutdown(signum, frame):
        print("[rover-control] shutting down (hardstop latch is file-backed "
              "and survives us)", flush=True)
        health.shutdown()
        sup.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    server.serve_forever()


if __name__ == "__main__":
    main()
