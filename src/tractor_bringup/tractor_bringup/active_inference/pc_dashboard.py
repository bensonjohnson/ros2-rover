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
  <div class="panel">
    <div style="font-size:12px;color:#7c8694;margin-bottom:6px;text-align:center">NEURAL NET ACTIVITY</div>
    <canvas id="brain" width="440" height="300"></canvas>
    <canvas id="heatmap" width="440" height="52" style="margin-top:4px"></canvas>
    <div class="legend">
      <span class="dot" style="background:#5bc0ff"></span>latent tanh
      <span class="dot" style="background:#ff5b5b"></span>state error
      <span class="dot" style="background:#9b59b6"></span>sensory error
    </div>
    <div style="display:flex;gap:16px;margin-top:5px;font-size:11px;color:#9aa4b1;">
      <span id="nn_status">-</span>
      <span>ensemble <span id="n_ens">5</span></span>
    </div>
  </div>
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

// ---- Neural Network Visualizer --------------------------------------------
function lerp3(a,b,t){return[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function heatColor(v){
  // v in [-1,1]: negative=blue, zero=dark, positive=orange/white
  if(v>0)return lerp3([30,20,10],[255,220,100],Math.min(1,v));
  return lerp3([5,5,20],[30,80,220],Math.min(1,-v));
}
function nodeColor(activation,err){
  // Nodes colored by tanh activation; ring glow by error magnitude.
  const r=Math.round((activation*0.5+0.5)*200);
  const g=Math.round((1-Math.abs(activation))*80);
  const b=Math.round(60);
  return[r,g,b];
}

function brainViz(s){
  const c=$('brain'),ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#1a1d23';ctx.fillRect(0,0,W,H);

  const sAct=s.s||[],zErr=s.e_z||[],obsErr=s.e_o||[];
  if(!sAct.length)return;

  // Layout: Latent layer (left) -> Decoder output (right)
  const lx=60,rx=W-60,cy=H/2;
  const cols=Math.ceil(Math.sqrt(sAct.length));
  const rows=Math.ceil(sAct.length/cols);
  const nr=Math.min(rows,8);           // cap at 8 rows for clarity
  const nc=Math.min(cols,Math.ceil(sAct.length/nr));
  const stepX=(rx-lx)/(nc+1),stepY=H/(nr+1);

  // Draw connection lines from latent to decoder (buses to observation bins).
  // Each latent node fans out to a representative obs-bin connection.
  const obsBins=Math.min(36,obsErr.length);
  const stepObs=Math.max(1,Math.floor(obsErr.length/obsBins));
  const connStep=Math.max(1,Math.floor(sAct.length/obsBins));
  for(let i=0;i<obsBins;i++){
    const li=Math.min(i*connStep,sAct.length-1);
    const ri=Math.min(i*stepObs,obsErr.length-1);
    const lnodex=lx+((li%nc)+1)*stepX;
    const lnodey=((Math.floor(li/nc)%nr)+1)*stepY;
    const rnodex=rx-Math.cos((ri/obsErr.length-0.5)*Math.PI)*(rx-lx)*0.35;
    const rnodey=cy+Math.sin((ri/obsErr.length-0.5)*Math.PI)*(H*0.42);
    const err=obsErr[ri];
    const[er,eg,eb]=heatColor(err);
    ctx.beginPath();ctx.moveTo(lnodex,lnodey);ctx.lineTo(rnodex,rnodey);
    ctx.strokeStyle=`rgba(${er},${eg},${eb},${Math.min(0.7,Math.abs(err)*1.5)})`;
    ctx.lineWidth=0.6;ctx.stroke();
  }

  // State prediction error arrows flowing into latent layer (left side).
  // Larger error = brighter red glow.
  if(zErr.length){
    const mxErr=Math.max(...zErr.map(Math.abs),0.001);
    for(let i=0;i<Math.min(sAct.length,nr*nc);i++){
      const lx2=lx+((i%nc)+1)*stepX;
      const ly2=((Math.floor(i/nc))+1)*stepY;
      const[,,eb]=heatColor(zErr[i]/mxErr);
      const rad=zErr[i]>0?4+Math.abs(zErr[i])*12:4;
      ctx.beginPath();ctx.arc(lx2,ly2,rad,0,2*Math.PI);
      ctx.fillStyle=`rgba(255,${Math.round(30-zErr[i]*40)},40,0.12)`;
      ctx.fill();
    }
  }

  // Draw latent nodes.
  const mxAct=Math.max(...sAct.map(Math.abs),0.001);
  for(let i=0;i<Math.min(sAct.length,nr*nc);i++){
    const nx=lx+((i%nc)+1)*stepX;
    const ny=((Math.floor(i/nc))+1)*stepY;
    const act=sAct[i]/mxAct;
    const[nr2,ng,nb]=nodeColor(act,0);
    const glow=Math.abs(act)*8;
    const[er2,eg2,eb2]=heatColor(zErr[i]||0);
    ctx.beginPath();ctx.arc(nx,ny,glow+5,0,2*Math.PI);
    ctx.fillStyle=`rgba(${er2},${eg2},10,0.25)`;ctx.fill();
    ctx.beginPath();ctx.arc(nx,ny,6,0,2*Math.PI);
    const grad=ctx.createRadialGradient(nx-1.5,ny-1.5,0,nx,ny,7);
    grad.addColorStop(0,`rgb(${Math.min(255,nr2+60)},${Math.min(255,ng+40)},${nb})`);
    grad.addColorStop(1,`rgb(${nr2},${ng},${nb})`);
    ctx.fillStyle=grad;ctx.fill();
    if(Math.abs(act)>0.3){ctx.strokeStyle='#fff';ctx.lineWidth=0.8;ctx.stroke();}
  }

  // Draw observation error nodes on the right (binned).
  for(let i=0;i<obsBins;i++){
    const ri=Math.min(i*stepObs,obsErr.length-1);
    const rx2=rx-Math.cos((ri/obsErr.length-0.5)*Math.PI)*(rx-lx)*0.35;
    const ry2=cy+Math.sin((ri/obsErr.length-0.5)*Math.PI)*(H*0.42);
    const[er3,eg3,eb3]=heatColor(obsErr[ri]);
    ctx.beginPath();ctx.arc(rx2,ry2,5,0,2*Math.PI);
    ctx.fillStyle=`rgba(${er3},${Math.min(255,eg3+60)},${eb3},0.85)`;ctx.fill();
    if(Math.abs(obsErr[ri])>0.15){ctx.strokeStyle='#fff';ctx.lineWidth=0.7;ctx.stroke();}
  }

  // Layer labels.
  ctx.font='10px system-ui';ctx.fillStyle='#5a6370';ctx.textAlign='center';
  ctx.fillText('z (latent)',lx,H-8);
  ctx.fillText('decoded',rx,H-8);
}

function heatmapViz(s){
  const c=$('heatmap'),ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#1a1d23';ctx.fillRect(0,0,W,H);
  const grid=s.W_o_grid||[];
  if(!grid.length)return;
  const n=grid.length;
  const cols=Math.min(64,Math.ceil(Math.sqrt(n*W/H)));
  const rows=Math.ceil(n/cols);
  const cw=W/cols,ch=H/rows;
  for(let i=0;i<n;i++){
    const col=i%cols,row=Math.floor(i/cols);
    const[er,eg,eb]=heatColor(grid[i]*3);   // scale weights for visibility
    ctx.fillStyle=`rgb(${er},${eg},${eb})`;
    ctx.fillRect(col*cw,row*ch,cw-0.4,ch-0.4);
  }
  ctx.font='9px system-ui';ctx.fillStyle='#5a6370';ctx.textAlign='left';
  ctx.fillText('W_o decoder weights (binned)',3,H-2);
}

function nnStatus(s){
  const n=s.n_ens||5;
  const te=s.trans_errors||[];
  const lx=60,rx=$('brain').width-60;
  $('n_ens').textContent=n;
  if(te.length){
    const sum=te.reduce((a,b)=>a+b,0)/te.length;
    const mx=Math.max(...te);const mn=Math.min(...te);
    $('nn_status').textContent=`trans err mean:${sum.toFixed(3)} range:[${mn.toFixed(3)},${mx.toFixed(3)}]`;
  }else{$('nn_status').textContent='waiting for data...';}
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
    radar(s);tracks(s);trace(s);brainViz(s);heatmapViz(s);nnStatus(s);
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

    def update(self, *, obs, pred, F, err, epi, epi_max, L, R, step,
                z=None, s=None, e_o=None, e_z=None, W_o=None,
                trans_errors=None) -> None:
        with self._lock:
            self._F.append(round(float(F), 4))
            self._err.append(round(float(err), 4))
            state: dict = {
                "obs": [round(float(x), 3) for x in obs],
                "pred": [round(float(x), 3) for x in pred],
                "F": float(F), "err": float(err),
                "epi": float(epi), "epi_max": float(epi_max),
                "L": float(L), "R": float(R), "step": int(step),
            }
            # Neural net activations & weights for the brain visualizer.
            if z is not None:
                state["z"] = [round(float(x), 4) for x in z]
            if s is not None:
                state["s"] = [round(float(x), 4) for x in s]
            if e_o is not None:
                state["e_o"] = [round(float(x), 4) for x in e_o]
            if e_z is not None:
                state["e_z"] = [round(float(x), 4) for x in e_z]
            if W_o is not None:
                # Flatten and bin into a coarse grid for the weight heatmap.
                # Show the top half of rows (most active decoder neurons).
                half = W_o.shape[0] // 2
                flat = W_o[:half].flatten()
                n = len(flat)
                bins = 256
                step_bin = max(1, n // bins)
                state["W_o_grid"] = [round(float(flat[i]), 4)
                                      for i in range(0, n, step_bin)]
            if trans_errors is not None:
                state["trans_errors"] = [round(float(x), 4) for x in trans_errors]
            self._state = state
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
