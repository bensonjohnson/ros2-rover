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
// Shows a 3-layer forward pass: observation (input) -> latent z -> prediction (output).
// - Input nodes (left): observed lidar, colored by magnitude.
// - Latent nodes (center): 64 nodes sized by |tanh activation|, colored by state error.
// - Output nodes (right): predicted observation, colored by prediction error.
// - Lines: thicker = more activation flowing through that path.

function lerp3(a,b,t){
  return[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];
}
// errorColor: 0=dim gray, high=orange/amber (easy to read against dark BG)
function errColor(v,mx){
  const t=mx>0?Math.min(1,Math.abs(v)/mx):0;
  // 0=dim, 1=amber/bright
  return[40+Math.round(t*215),20+Math.round(t*180),10];
}
// activationColor: negative=blue, zero=gray, positive=cyan-green
function actColor(v,mx){
  const t=mx>0?Math.min(1,Math.abs(v)/mx):0;
  if(v>=0)return lerp3([60,60,60],[0,200,140],t);
  return lerp3([60,60,60],[40,100,220],t);
}
// heatColor: for weight heatmap — blue=negative, orange=positive
function heatColor(v){
  const t=Math.max(-1,Math.min(1,v));
  if(t>0)return lerp3([40,25,12],[255,210,80],t);
  return lerp3([40,25,12],[60,120,230],-t);
}

function brainViz(s){
  const c=$('brain'),ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#14171d';ctx.fillRect(0,0,W,H);

  const obs=s.obs||[],sAct=s.s||[],pred=s.pred||[],zAbs=s.z_abs||[];
  const eZ=s.e_z_abs||[],eO=s.e_o||[],predErrArr=pred.map((v,i)=>Math.abs(eO[i]||0));
  if(!sAct.length)return;

  const MARGIN=28,TOP=18,BOT=H-12,ly=MARGIN,ry=H-MARGIN;
  const lCol=W*0.2,lMid=W*0.5,rCol=W*0.8;

  // ── 1. Column labels ────────────────────────────────────────────────
  ctx.font='bold 11px system-ui';ctx.fillStyle='#8892a0';ctx.textAlign='center';
  ctx.fillText('OBSERVATION',lCol,TOP+8);
  ctx.fillText('LATENT  z',lMid,TOP+8);
  ctx.fillText('PREDICTION',rCol,TOP+8);

  // ── 2. Build node grids ─────────────────────────────────────────────
  // Number of displayed nodes (space-constrained; sample from full arrays).
  const N_DISP=Math.min(20,obs.length,pred.length);
  const obsDispStep=Math.max(1,Math.floor(obs.length/N_DISP));
  const predDispStep=Math.max(1,Math.floor(pred.length/N_DISP));

  const obsY=[],predY=[];
  for(let i=0;i<N_DISP;i++){
    const t=(i+0.5)/N_DISP;
    obsY.push(ly+t*(ry-ly));
    predY.push(ly+t*(ry-ly));
  }

  const zNodes=Math.min(sAct.length,zAbs.length);
  const zCols=8,zRows=Math.ceil(zNodes/zCols);
  const zX=[],zY=[];
  const zStepY=(ry-ly)/(zRows+1);
  const zSpan=lMid-lCol-20;
  for(let r=0;r<zRows;r++){
    for(let c2=0;c2<zCols;c2++){
      const idx=r*zCols+c2;if(idx>=zNodes)break;
      zX.push(lMid+(c2-(zCols-1)*0.5)*(zSpan/Math.max(1,zCols-1)));
      zY.push(ly+(r+1)*zStepY);
    }
  }

  // ── 3. Connections: latent (col 2) -> prediction (col 3) ───────────
  // Sort all z->pred connections by pred error magnitude.
  // Draw top-K so the canvas stays readable.
  const connK=Math.min(120,zX.length*predY.length);
  const conns=[];
  const mxPE=Math.max(...predErrArr,0.001);
  for(let zi=0;zi<zX.length;zi++){
    for(let pi=0;pi<predY.length;pi++){
      conns.push({zi,pi,str:predErrArr[pi]/mxPE});
    }
  }
  // Draw strongest connections first; weaker ones on top with lower alpha.
  conns.sort((a,b)=>b.str-a.str);
  for(let i=0;i<connK;i++){
    const{zi,pi,str}=conns[i];
    const[er,eg,eb]=errColor(predErrArr[pi],mxPE);
    ctx.beginPath();
    ctx.moveTo(zX[zi]+4,zY[zi]);
    ctx.lineTo(rCol-10,predY[pi]);
    ctx.strokeStyle=`rgba(${er},${eg},${eb},${0.06+str*0.5})`;
    ctx.lineWidth=0.35+str*1.6;
    ctx.stroke();
  }

  // ── 4. Observation nodes (col 1) ────────────────────────────────────
  const mxObs=Math.max(...obs,0.001);
  obsY.forEach((ny,i)=>{
    const v=obs[Math.min(i*obsDispStep,obs.length-1)]||0;
    const br=0.35+0.65*(v/mxObs);
    ctx.beginPath();ctx.arc(lCol,ny,3.5,0,2*Math.PI);
    ctx.fillStyle=`rgba(${Math.round(90*br)},${Math.round(90*br)},${Math.round(90*br)},0.9)`;
    ctx.fill();
  });

  // ── 5. Latent nodes (col 2) ─────────────────────────────────────────
  const mxZ=Math.max(...zAbs,0.001);
  zX.forEach((nx,zi)=>{
    const ny=zY[zi];
    const absAct=zAbs[zi]||0;
    const activation=sAct[zi]||0;
    const radius=3+absAct*7;
    const[ar,ag,ab]=actColor(activation,mxZ);
    // glow
    ctx.beginPath();ctx.arc(nx,ny,radius+absAct*6,0,2*Math.PI);
    ctx.fillStyle=`rgba(${ar},${ag},${ab},0.13)`;ctx.fill();
    // core
    ctx.beginPath();ctx.arc(nx,ny,radius,0,2*Math.PI);
    ctx.fillStyle=`rgb(${ar},${ag},${ab})`;ctx.fill();
    if(absAct>mxZ*0.5){ctx.strokeStyle='rgba(255,255,255,0.65)';ctx.lineWidth=0.9;ctx.stroke();}
  });

  // ── 6. Prediction nodes (col 3) ─────────────────────────────────────
  const mxPE2=Math.max(...predErrArr,0.001);
  predY.forEach((ny,i)=>{
    const pi=Math.min(i*predDispStep,pred.length-1);
    const err=Math.abs(eO[pi]||0);
    const v=pred[pi]||0;
    const[er,eg,eb]=errColor(err,mxPE2);
    const bright=0.3+0.7*Math.abs(v);
    ctx.beginPath();ctx.arc(rCol,ny,3.5,0,2*Math.PI);
    ctx.fillStyle=`rgba(${Math.round(er*bright+30*(1-bright))},${Math.round(eg*bright+30*(1-bright))},${Math.round(eb*bright+30*(1-bright))},0.92)`;
    ctx.fill();
    if(err>mxPE2*0.4){ctx.strokeStyle='#fff';ctx.lineWidth=0.8;ctx.stroke();}
  });

  // ── 7. Column dividers & arrows ─────────────────────────────────────
  ctx.setLineDash([4,4]);ctx.strokeStyle='#252a35';ctx.lineWidth=1;
  [lCol+14,rCol-14].forEach(x=>{
    ctx.beginPath();ctx.moveTo(x,TOP);ctx.lineTo(x,BOT);ctx.stroke();
  });ctx.setLineDash([]);
  // Flow arrows between cols
  const ay=(ly+ry)/2;
  ctx.fillStyle='#4ab8ff';
  ctx.beginPath();ctx.moveTo(lCol+14,ay);ctx.lineTo(lCol+28,ay-4);ctx.lineTo(lCol+28,ay+4);
  ctx.closePath();ctx.fill();
  ctx.beginPath();ctx.moveTo(rCol-14,ay);ctx.lineTo(rCol-28,ay-4);ctx.lineTo(rCol-28,ay+4);
  ctx.closePath();ctx.fill();

  // ── 8. Error legend ─────────────────────────────────────────────────
  const barW=90,barH=6,barX=W-barW-12,barY=BOT-10;
  const grd=ctx.createLinearGradient(barX,0,barX+barW,0);
  grd.addColorStop(0,'#1e1408');grd.addColorStop(1,'#ffc864');
  ctx.fillStyle=grd;ctx.fillRect(barX,barY,barW,barH);
  ctx.font='9px system-ui';ctx.fillStyle='#5a6370';ctx.textAlign='left';
  ctx.fillText('error: dim=correct  bright=wrong',barX,barY-2);

  // ── 9. Stats overlay ────────────────────────────────────────────────
  const mxEZ=Math.max(...eZ,0.001);
  const avgEZ=eZ.length?eZ.reduce((a,b)=>a+b,0)/eZ.length:0;
  ctx.font='10px system-ui';ctx.fillStyle='#6a7485';ctx.textAlign='right';
  ctx.fillText(`z err avg:${avgEZ.toFixed(3)} max:${mxEZ.toFixed(3)}`,W-12,TOP+8);
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
  const te=s.trans_errors||[];
  if(te.length){
    const sum=te.reduce((a,b)=>a+b,0)/te.length;
    const mx=Math.max(...te);const mn=Math.min(...te);
    $('nn_status').textContent=`ensemble=${te.length}  trans err μ=${sum.toFixed(3)}  range=[${mn.toFixed(3)},${mx.toFixed(3)}]`;
  }else{$('nn_status').textContent='ensemble=5  trans errors: connecting...';}
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
                trans_errors=None, z_abs=None, e_z_abs=None) -> None:
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
                state["W_o_grid"] = [round(float(x), 4)
                                      for x in W_o.flatten()[::4].tolist()]  # every 4th
            if z is not None and s is not None:
                # |tanh activation| per latent node: drives node size in the diagram.
                state["z_abs"] = [round(float(abs(x)), 4) for x in s]
            if trans_errors is not None:
                state["trans_errors"] = [round(float(x), 4) for x in trans_errors]
            if e_z_abs is not None:
                state["e_z_abs"] = [round(float(x), 4) for x in e_z_abs]
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
