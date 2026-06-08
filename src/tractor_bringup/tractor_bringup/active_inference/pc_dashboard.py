"""Rover-hosted dashboard for the predictive-coding brain.

Serves a single page that polls a JSON `/state` endpoint and renders, on a
canvas:
  - a radial "openness map": the observed lidar (what IS) overlaid with the
    brain's reconstruction (what it EXPECTED) — the gap between the two rings
    is the sensory prediction error you're watching shrink.
  - live traces of free energy and sensory error.
  - the current track command (L/R bars) and epistemic value.

Threaded http.server only — no Flask/Tornado dependency, matching the existing
nomad_dashboard.py / RLPD dashboard pattern. The node pushes a fresh state dict
each control tick; the server keeps short ring buffers for the traces so a
freshly opened page immediately shows recent history.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>PNN Brain</title>
<style>
  body{background:#0d0f12;color:#cdd3da;font-family:system-ui,sans-serif;margin:0;
       display:flex;flex-direction:column;align-items:center;}
  h1{font-size:15px;font-weight:600;margin:12px 0 2px;letter-spacing:.06em;}
  .sub{font-size:12px;color:#7c8694;margin-bottom:8px;}
  .wrap{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;align-items:flex-start;}
  .panel{background:#15181d;border:1px solid #232830;border-radius:8px;padding:10px;}
  canvas{display:block;}
  .stats{font-size:12px;line-height:1.7;min-width:150px;}
  .k{color:#7c8694;} .v{color:#e8edf2;font-variant-numeric:tabular-nums;}
  .legend{font-size:11px;color:#9aa4b1;margin-top:6px;}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle;}
  .stale{color:#ff5b5b;}
</style></head><body>
<h1>PREDICTIVE-CODING ROVER BRAIN</h1>
<div class="sub">observed vs. predicted lidar &mdash; pure epistemic active inference</div>
<div class="wrap">
  <div class="panel"><canvas id="radar" width="360" height="360"></canvas>
    <div class="legend"><span class="dot" style="background:#39ff14"></span>observed
      <span class="dot" style="background:#ff9d3b"></span>predicted</div></div>
  <div class="panel stats">
    <div><span class="k">step</span> <span class="v" id="step">-</span></div>
    <div><span class="k">free energy</span> <span class="v" id="F">-</span></div>
    <div><span class="k">sensory err</span> <span class="v" id="err">-</span></div>
    <div><span class="k">epistemic</span> <span class="v" id="epi">-</span></div>
    <div><span class="k">epi max</span> <span class="v" id="epimax">-</span></div>
    <div style="margin-top:8px"><span class="k">track L</span> <span class="v" id="L">-</span>
       &nbsp; <span class="k">R</span> <span class="v" id="R">-</span></div>
    <canvas id="tracks" width="150" height="60" style="margin-top:6px"></canvas>
    <div id="status" class="legend"></div>
  </div>
  <div class="panel"><canvas id="trace" width="320" height="200"></canvas>
    <div class="legend"><span class="dot" style="background:#5bc0ff"></span>free energy
      <span class="dot" style="background:#ff9d3b"></span>sensory err</div></div>
</div>
<script>
const $=id=>document.getElementById(id);
function polar(cx,cy,r,a){return [cx+r*Math.cos(a),cy+r*Math.sin(a)];}
function drawRing(ctx,cx,cy,R,arr,color){
  if(!arr||!arr.length)return;
  ctx.beginPath();
  for(let i=0;i<arr.length;i++){
    // Top-down view: forward (scan 0deg) = up; angle increases CCW so the
    // rover's left renders on the left. (negate to undo canvas Y-down mirror)
    const a=-(i/arr.length)*2*Math.PI - Math.PI/2;
    const [x,y]=polar(cx,cy,arr[i]*R,a);
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
  }
  ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
}
function radar(s){
  const c=$('radar'),ctx=c.getContext('2d'),W=c.width,cx=W/2,cy=W/2,R=W/2-12;
  ctx.clearRect(0,0,W,W);
  ctx.strokeStyle='#2a3038';ctx.lineWidth=1;
  for(const f of [0.33,0.66,1.0]){ctx.beginPath();ctx.arc(cx,cy,f*R,0,2*Math.PI);ctx.stroke();}
  drawRing(ctx,cx,cy,R,s.pred,'#ff9d3b');
  drawRing(ctx,cx,cy,R,s.obs,'#39ff14');
  ctx.fillStyle='#cdd3da';ctx.beginPath();ctx.arc(cx,cy,4,0,2*Math.PI);ctx.fill();
  // forward marker (rover faces up)
  ctx.strokeStyle='#5bc0ff';ctx.lineWidth=2;ctx.beginPath();
  ctx.moveTo(cx,cy);ctx.lineTo(cx,cy-18);ctx.stroke();
  ctx.fillStyle='#5bc0ff';ctx.beginPath();
  ctx.moveTo(cx,cy-22);ctx.lineTo(cx-4,cy-14);ctx.lineTo(cx+4,cy-14);ctx.closePath();ctx.fill();
  ctx.font='10px system-ui';ctx.fillStyle='#7c8694';ctx.textAlign='center';
  ctx.fillText('FWD',cx,cy-26);ctx.textAlign='left';
}
function tracks(s){
  const c=$('tracks'),ctx=c.getContext('2d'),W=c.width,H=c.height,mid=H/2;
  ctx.clearRect(0,0,W,H);ctx.strokeStyle='#2a3038';ctx.beginPath();
  ctx.moveTo(0,mid);ctx.lineTo(W,mid);ctx.stroke();
  const bar=(x,v)=>{ctx.fillStyle=v>=0?'#39ff14':'#ff5b5b';
    const h=v*(mid-4);ctx.fillRect(x,mid,40,-h);};
  ctx.font='10px system-ui';ctx.fillStyle='#7c8694';
  bar(25,s.L||0);bar(85,s.R||0);
}
let Fh=[],Eh=[];
function trace(s){
  if(s.F_hist){Fh=s.F_hist;Eh=s.err_hist;}
  const c=$('trace'),ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  const plot=(arr,color)=>{if(!arr.length)return;
    const mx=Math.max(...arr,1e-6),mn=Math.min(...arr);
    const rng=(mx-mn)||1;ctx.beginPath();
    for(let i=0;i<arr.length;i++){const x=i/(arr.length-1||1)*W;
      const y=H-((arr[i]-mn)/rng)*(H-10)-5;i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();};
  plot(Fh,'#5bc0ff');plot(Eh,'#ff9d3b');
}
async function tick(){
  try{
    const s=await (await fetch('/state')).json();
    const fix=(x,n=3)=>(x==null?'-':x.toFixed(n));
    $('step').textContent=s.step??'-';
    $('F').textContent=fix(s.F);$('err').textContent=fix(s.err);
    $('epi').textContent=fix(s.epi,4);$('epimax').textContent=fix(s.epi_max,4);
    $('L').textContent=fix(s.L,2);$('R').textContent=fix(s.R,2);
    const age=s.age??99;
    $('status').textContent=age>1.5?'⚠ stale ('+age.toFixed(1)+'s)':'live';
    $('status').className='legend'+(age>1.5?' stale':'');
    radar(s);tracks(s);trace(s);
  }catch(e){$('status').textContent='disconnected';}
}
setInterval(tick,100);tick();
</script></body></html>
"""


class PCDashboardState:
    """Thread-safe holder for the latest brain state + short trace history."""

    def __init__(self, history: int = 240):
        self._lock = threading.Lock()
        self._state: dict = {}
        self._stamp = 0.0
        self._F = deque(maxlen=history)
        self._err = deque(maxlen=history)

    def update(self, *, obs, pred, F, err, epi, epi_max, L, R, step) -> None:
        with self._lock:
            self._F.append(round(float(F), 4))
            self._err.append(round(float(err), 4))
            self._state = {
                "obs": [round(float(x), 3) for x in obs],
                "pred": [round(float(x), 3) for x in pred],
                "F": float(F), "err": float(err),
                "epi": float(epi), "epi_max": float(epi_max),
                "L": float(L), "R": float(R), "step": int(step),
            }
            self._stamp = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
            if s:
                s["age"] = time.monotonic() - self._stamp
                s["F_hist"] = list(self._F)
                s["err_hist"] = list(self._err)
            return s


def _make_handler(state: PCDashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body = json.dumps(state.snapshot()).encode()
                ctype = "application/json"
            else:
                body = _PAGE.encode()
                ctype = "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return Handler


def start_dashboard_server(state: PCDashboardState, port: int = 8082) -> ThreadingHTTPServer:
    """Start the dashboard HTTP server (retries bind while a prior run frees the port)."""
    handler = _make_handler(state)
    last_err: OSError | None = None
    for _ in range(12):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), handler)
            break
        except OSError as exc:
            last_err = exc
            time.sleep(0.5)
    else:
        raise last_err if last_err else OSError(f"could not bind port {port}")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
