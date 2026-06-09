"""Dashboard for the growing-latent-field cognitive-map brain.

Adds a top-down MAP panel to the openness view: every discovered cell is drawn,
coloured by its predicted openness and faded by confidence, with frontier cells
outlined and the rover drawn at its odom pose. Watching this panel is watching
the map extend and fill in as the rover explores.

Same threaded http.server / JSON-poll pattern as pc_dashboard.py.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>PNN Cognitive Map</title>
<style>
  body{
    background: radial-gradient(circle at center, #141923 0%, #0d0f12 100%);
    color:#cdd3da;
    font-family: 'Outfit', system-ui, sans-serif;
    margin:0;
    display:flex;
    flex-direction:column;
    align-items:center;
    min-height: 100vh;
    padding-bottom: 30px;
  }
  h1{
    font-size:22px;
    font-weight:700;
    margin:24px 0 4px;
    letter-spacing:.04em;
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .sub{
    font-size:13px;
    color:#7c8694;
    margin-bottom:20px;
    font-weight: 300;
  }
  .wrap{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;align-items:stretch;}
  .panel{
    background: rgba(21, 24, 29, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius:14px;
    padding:16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
  }
  canvas{display:block; border-radius: 8px;}
  .legend{font-size:11px;color:#8c97a5;margin-top:10px;line-height:1.6;}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle;}
  
  /* Telemetry Panel */
  .stats-panel {
    min-width: 280px;
    justify-content: space-between;
  }
  .stats-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 10px;
    margin-bottom: 12px;
  }
  .stats-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #7c8694;
  }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 16px;
  }
  .stat-card {
    background: rgba(30, 34, 43, 0.5);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 8px 12px;
  }
  .stat-card.full-width {
    grid-column: span 2;
  }
  .stat-label {
    font-size: 9px;
    font-weight: 600;
    color: #5a6370;
    letter-spacing: 0.05em;
  }
  .stat-value {
    font-size: 18px;
    font-weight: 700;
    color: #e8edf2;
    margin-top: 4px;
    font-variant-numeric: tabular-nums;
  }
  .stat-card.color-blue .stat-value { color: #5bc0ff; }
  .stat-card.color-orange .stat-value { color: #ff9d3b; }
  .stat-card.color-gold .stat-value { color: #ffd043; }
  .stat-card.color-green .stat-value { color: #00c88c; }
  
  /* Track Controls */
  .tracks-panel {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 12px;
    margin-bottom: 12px;
  }
  .track-bar-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
  }
  .track-bar-wrapper .k {
    font-size: 10px;
    font-weight: 700;
    color: #7c8694;
    width: 60px;
  }
  .track-gauge {
    position: relative;
    flex-grow: 1;
    height: 12px;
    background: #15181d;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.04);
    overflow: hidden;
  }
  .track-center {
    position: absolute;
    left: 50%;
    top: 0;
    width: 2px;
    height: 100%;
    background: rgba(255,255,255,0.25);
    z-index: 2;
  }
  .track-fill {
    height: 100%;
    width: 0%;
    transition: width 0.05s ease;
  }
  .fill-neg {
    position: absolute;
    right: 50%;
    background: linear-gradient(90deg, #ff5b5b, #ff8c8c);
    border-radius: 6px 0 0 6px;
  }
  .fill-pos {
    position: absolute;
    left: 50%;
    background: linear-gradient(90deg, #00c88c, #39ff14);
    border-radius: 0 6px 6px 0;
  }
  .track-bar-wrapper .v {
    font-size: 13px;
    font-weight: 600;
    color: #e8edf2;
    width: 40px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  
  /* Status Badge */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .status-badge.live {
    background: rgba(0, 200, 140, 0.1);
    border: 1px solid rgba(0, 200, 140, 0.25);
    color: #00c88c;
  }
  .status-badge.stale {
    background: rgba(255, 91, 91, 0.1);
    border: 1px solid rgba(255, 91, 91, 0.25);
    color: #ff5b5b;
  }
  .status-badge.done {
    background: rgba(0, 200, 140, 0.15);
    border: 1px solid rgba(0, 200, 140, 0.3);
    color: #39ff14;
    font-weight: bold;
  }
  .status-badge.disconnected {
    background: rgba(124, 134, 148, 0.1);
    border: 1px solid rgba(124, 134, 148, 0.25);
    color: #7c8694;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 6px currentColor;
  }
  .status-badge.live .status-dot {
    animation: pulse 1.6s infinite ease-in-out;
  }
  @keyframes pulse {
    0% { transform: scale(0.9); opacity: 0.5; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.5; }
  }
</style></head><body>
<h1>PNN COGNITIVE MAP &mdash; growing latent field</h1>
<div class="sub">the rover builds &amp; fills in an allocentric map as it explores</div>
<div class="wrap">
  <div class="panel"><canvas id="map" width="420" height="420"></canvas>
    <div class="legend">brightness = openness &nbsp; faint = low confidence
      <span class="dot" style="background:#ff9d3b"></span>frontier
      <span class="dot" style="background:#5bc0ff"></span>rover</div></div>
  <div class="panel"><canvas id="radar" width="240" height="240"></canvas>
    <div class="legend"><span class="dot" style="background:#39ff14"></span>observed
      <span class="dot" style="background:#ff9d3b"></span>predicted</div></div>
      
  <div class="panel stats-panel">
    <div class="stats-header">
      <div class="stats-title">TELEMETRY</div>
      <div class="status-badge live" id="status_badge">
        <span class="status-dot"></span>
        <span id="status">live</span>
      </div>
    </div>
    
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">STEP</div>
        <div class="stat-value" id="step">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">CELLS MAPPED</div>
        <div class="stat-value" id="cells">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">FRONTIERS</div>
        <div class="stat-value" id="front">-</div>
      </div>
      <div class="stat-card color-orange">
        <div class="stat-label">DECODE ERR</div>
        <div class="stat-value" id="err">-</div>
      </div>
      <div class="stat-card color-gold">
        <div class="stat-label">NOVELTY</div>
        <div class="stat-value" id="nov">-</div>
      </div>
      <div class="stat-card full-width color-blue">
        <div class="stat-label">POSE</div>
        <div class="stat-value" id="pose" style="font-size: 14px; margin-top: 6px;">-</div>
      </div>
    </div>
    
    <div class="tracks-panel">
      <div class="track-bar-wrapper">
        <span class="k">TRACK L</span>
        <div class="track-gauge">
          <div class="track-fill fill-neg" id="l_neg"></div>
          <div class="track-center"></div>
          <div class="track-fill fill-pos" id="l_pos"></div>
        </div>
        <span class="v" id="L">-</span>
      </div>
      <div class="track-bar-wrapper">
        <span class="k">TRACK R</span>
        <div class="track-gauge">
          <div class="track-fill fill-neg" id="r_neg"></div>
          <div class="track-center"></div>
          <div class="track-fill fill-pos" id="r_pos"></div>
        </div>
        <span class="v" id="R">-</span>
      </div>
    </div>
    
    <canvas id="trace" width="150" height="80" style="margin-top:8px"></canvas>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
function radar(s){
  const c=$('radar'),x=c.getContext('2d'),W=c.width,cx=W/2,cy=W/2,R=W/2-10;
  x.clearRect(0,0,W,W);x.strokeStyle='#2a3038';x.lineWidth=1;
  for(const f of[0.5,1.0]){x.beginPath();x.arc(cx,cy,f*R,0,2*Math.PI);x.stroke();}
  const ring=(arr,col)=>{if(!arr||!arr.length)return;x.beginPath();
    for(let i=0;i<arr.length;i++){const a=-(i/arr.length)*2*Math.PI-Math.PI/2;
      const px=cx+arr[i]*R*Math.cos(a),py=cy+arr[i]*R*Math.sin(a);i?x.lineTo(px,py):x.moveTo(px,py);}
    x.closePath();x.strokeStyle=col;x.lineWidth=2;x.stroke();};
  ring(s.pred,'#ff9d3b');ring(s.obs,'#39ff14');
  x.fillStyle='#5bc0ff';x.beginPath();x.moveTo(cx,cy-R*0.12);x.lineTo(cx-4,cy);x.lineTo(cx+4,cy);x.closePath();x.fill();
}
function drawMap(s){
  const c=$('map'),x=c.getContext('2d'),W=c.width;x.clearRect(0,0,W,W);
  const cells=s.cells||[];
  if(!cells.length){return;}
  let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
  for(const [ix,iy] of cells){minx=Math.min(minx,ix);maxx=Math.max(maxx,ix);miny=Math.min(miny,iy);maxy=Math.max(maxy,iy);}
  // include rover cell in bounds
  const rcx=Math.floor(s.x/s.cell),rcy=Math.floor(s.y/s.cell);
  minx=Math.min(minx,rcx);maxx=Math.max(maxx,rcx);miny=Math.min(miny,rcy);maxy=Math.max(maxy,rcy);
  const span=Math.max(maxx-minx,maxy-miny,1)+2, sc=W/span;
  const X=ix=>(ix-minx+1)*sc, Y=iy=>W-(iy-miny+1)*sc;  // y up
  for(const [ix,iy,conf,open] of cells){
    const g=Math.round(40+open*200), a=0.25+0.75*conf;
    x.fillStyle='rgba('+Math.round(g*0.5)+','+g+','+Math.round(g*0.6)+','+a+')';
    x.fillRect(X(ix),Y(iy)-sc,Math.ceil(sc),Math.ceil(sc));
  }
  for(const [ix,iy] of (s.frontiers||[])){
    x.strokeStyle='#ff9d3b';x.lineWidth=1.5;x.strokeRect(X(ix)+0.5,Y(iy)-sc+0.5,sc-1,sc-1);
  }
  // rover
  const px=X(rcx)+sc/2,py=Y(rcy)-sc/2,th=s.theta||0;
  x.fillStyle='#5bc0ff';x.beginPath();
  x.moveTo(px+9*Math.cos(th),py-9*Math.sin(th));
  x.lineTo(px+6*Math.cos(th+2.5),py-6*Math.sin(th+2.5));
  x.lineTo(px+6*Math.cos(th-2.5),py-6*Math.sin(th-2.5));
  x.closePath();x.fill();
}
let H=[];
function trace(s){
  if(s.err_hist)H=s.err_hist;
  const c=$('trace'),x=c.getContext('2d'),W=c.width,Ht=c.height;x.clearRect(0,0,W,Ht);
  if(!H.length)return;const mx=Math.max(...H,1e-6),mn=Math.min(...H),rg=(mx-mn)||1;
  x.beginPath();for(let i=0;i<H.length;i++){const xx=i/(H.length-1||1)*W,yy=Ht-((H[i]-mn)/rg)*(Ht-8)-4;i?x.lineTo(xx,yy):x.moveTo(xx,yy);}
  x.strokeStyle='#ff9d3b';x.lineWidth=1.5;x.stroke();
}
function tracks(s){
  const L = s.L || 0;
  const R = s.R || 0;
  const lNeg = $('l_neg'), lPos = $('l_pos');
  const rNeg = $('r_neg'), rPos = $('r_pos');
  if (lNeg && lPos) {
    if (L >= 0) {
      lPos.style.width = (L * 50) + '%';
      lNeg.style.width = '0%';
    } else {
      lNeg.style.width = (Math.abs(L) * 50) + '%';
      lPos.style.width = '0%';
    }
  }
  if (rNeg && rPos) {
    if (R >= 0) {
      rPos.style.width = (R * 50) + '%';
      rNeg.style.width = '0%';
    } else {
      rNeg.style.width = (Math.abs(R) * 50) + '%';
      rPos.style.width = '0%';
    }
  }
}

async function tick(){
  try{
    const s=await(await fetch('/state')).json();
    const f=(v,n=2)=>v==null?'-':v.toFixed(n);
    $('step').textContent=s.step??'-';
    $('cells').textContent=s.cells?s.cells.length:0;
    $('front').textContent=(s.frontiers||[]).length;
    $('err').textContent=f(s.err,3);
    $('nov').textContent=f(s.nov,3);
    $('L').textContent=f(s.L);
    $('R').textContent=f(s.R);
    $('pose').textContent=f(s.x)+', '+f(s.y)+' @'+f((s.theta||0)*57.3,0)+'°';
    
    const age=s.age??99;
    const badge = $('status_badge');
    const statusText = $('status');
    if (s.done) {
      badge.className = 'status-badge done';
      statusText.textContent = 'exploration complete';
    } else if (age > 1.5) {
      badge.className = 'status-badge stale';
      statusText.textContent = 'stale (' + age.toFixed(1) + 's)';
    } else {
      badge.className = 'status-badge live';
      statusText.textContent = 'live';
    }
    
    drawMap(s);
    radar(s);
    trace(s);
    tracks(s);
  }catch(e){
    const badge = $('status_badge');
    const statusText = $('status');
    if (badge) badge.className = 'status-badge disconnected';
    if (statusText) statusText.textContent = 'disconnected';
  }
}
setInterval(tick,150);tick();
</script></body></html>
"""


class MapDashboardState:
    def __init__(self, history: int = 240):
        self._lock = threading.Lock()
        self._state: dict = {}
        self._stamp = 0.0
        self._err = deque(maxlen=history)

    def update(self, *, obs, pred, err, nov, cells, frontiers, x, y, theta,
               cell_size, L, R, step, done) -> None:
        with self._lock:
            self._err.append(round(float(err), 4))
            self._state = {
                "obs": [round(float(v), 3) for v in obs],
                "pred": [round(float(v), 3) for v in pred],
                "err": float(err), "nov": float(nov),
                "cells": [[ix, iy, round(c, 3), round(o, 3)] for (ix, iy, c, o) in cells],
                "frontiers": [[ix, iy] for (ix, iy) in frontiers],
                "x": float(x), "y": float(y), "theta": float(theta),
                "cell": float(cell_size), "L": float(L), "R": float(R),
                "step": int(step), "done": bool(done),
            }
            self._stamp = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
            if s:
                s["age"] = time.monotonic() - self._stamp
                s["err_hist"] = list(self._err)
            return s


def _make_handler(state: MapDashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body = json.dumps(state.snapshot()).encode(); ctype = "application/json"
            else:
                body = _PAGE.encode(); ctype = "text/html"
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


def start_dashboard_server(state: MapDashboardState, port: int = 8083) -> ThreadingHTTPServer:
    handler = _make_handler(state)
    last_err = None
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
