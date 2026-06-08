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
  body{background:#0d0f12;color:#cdd3da;font-family:system-ui,sans-serif;margin:0;
       display:flex;flex-direction:column;align-items:center;}
  h1{font-size:15px;font-weight:600;margin:12px 0 2px;letter-spacing:.06em;}
  .sub{font-size:12px;color:#7c8694;margin-bottom:8px;}
  .wrap{display:flex;gap:16px;flex-wrap:wrap;justify-content:center;align-items:flex-start;}
  .panel{background:#15181d;border:1px solid #232830;border-radius:8px;padding:10px;}
  .stats{font-size:12px;line-height:1.7;min-width:150px;}
  .k{color:#7c8694;} .v{color:#e8edf2;font-variant-numeric:tabular-nums;}
  .legend{font-size:11px;color:#9aa4b1;margin-top:6px;}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle;}
  .stale{color:#ff5b5b;} .done{color:#39ff14;font-weight:600;}
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
  <div class="panel stats">
    <div><span class="k">step</span> <span class="v" id="step">-</span></div>
    <div><span class="k">cells mapped</span> <span class="v" id="cells">-</span></div>
    <div><span class="k">frontiers</span> <span class="v" id="front">-</span></div>
    <div><span class="k">decode err</span> <span class="v" id="err">-</span></div>
    <div><span class="k">novelty</span> <span class="v" id="nov">-</span></div>
    <div style="margin-top:6px"><span class="k">pose</span> <span class="v" id="pose">-</span></div>
    <div><span class="k">track L</span> <span class="v" id="L">-</span> <span class="k">R</span> <span class="v" id="R">-</span></div>
    <canvas id="trace" width="150" height="80" style="margin-top:8px"></canvas>
    <div id="status" class="legend"></div>
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
async function tick(){
  try{
    const s=await(await fetch('/state')).json();
    const f=(v,n=2)=>v==null?'-':v.toFixed(n);
    $('step').textContent=s.step??'-';$('cells').textContent=s.cells?s.cells.length:0;
    $('front').textContent=(s.frontiers||[]).length;$('err').textContent=f(s.err,3);
    $('nov').textContent=f(s.nov,3);$('L').textContent=f(s.L);$('R').textContent=f(s.R);
    $('pose').textContent=f(s.x)+', '+f(s.y)+' @'+f((s.theta||0)*57.3,0)+'°';
    const age=s.age??99;
    $('status').textContent=s.done?'✓ exploration complete':(age>1.5?'⚠ stale '+age.toFixed(1)+'s':'live');
    $('status').className='legend'+(s.done?' done':(age>1.5?' stale':''));
    drawMap(s);radar(s);trace(s);
  }catch(e){$('status').textContent='disconnected';}
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
