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
  .brain-row{display:flex;justify-content:center;margin-bottom:20px;}
  .brain-panel{
    background: rgba(21, 24, 29, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius:14px;
    padding:16px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
  }
  .brain-title{
    font-size:12px;
    font-weight:700;
    color:#7c8a9a;
    letter-spacing:.15em;
    text-transform:uppercase;
    text-align:center;
    margin-bottom:8px;
  }
  .brain-sub{font-size:11px;color:#5a6370;text-align:center;margin-bottom:12px;}
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
  
  /* Track Controls */
  .tracks-panel {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 12px;
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

  .legend {
    font-size: 11px;
    color: #8c97a5;
    margin-top: 10px;
    line-height: 1.6;
  }
  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin: 0 4px 0 10px;
    vertical-align: middle;
  }
  .nn-row {
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: #6a7482;
    margin-top: 12px;
    flex-wrap: wrap;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 10px;
  }
  .nn-stat { display: flex; gap: 4px; align-items: center; }
  .nn-stat-val { color: #b0bac5; font-variant-numeric: tabular-nums; }
  .bar-wrap { display: inline-flex; align-items: center; gap: 6px; }
  .bar-bg { background: #15181d; border-radius: 3px; display: inline-block; width: 80px; height: 6px; overflow: hidden; }
  .err-bar { height: 100%; border-radius: 3px; display: block; }
</style></head><body>
<h1>PREDICTIVE-CODING ROVER BRAIN</h1>
<div class="sub">observed vs. predicted lidar &mdash; pure epistemic active inference</div>
<div class="brain-row">
  <div class="brain-panel">
    <div class="brain-title">NEURAL NET ACTIVITY</div>
    <div class="brain-sub">forward pass: observation &rarr; latent z &rarr; prediction</div>
    <canvas id="brain" width="920" height="380"></canvas>
    <div class="legend" style="margin-top:6px;text-align:center;">
      <span class="dot" style="background:#5bc0ff"></span>latent active
      <span class="dot" style="background:#3060d0"></span>latent suppressed
      <span class="dot" style="background:#ff9d3b"></span>pred error (bright=wrong)
      <span class="dot" style="background:#00c88c"></span>positive flow
      <span class="dot" style="background:#ff6432"></span>negative flow
    </div>
    <div class="nn-row">
      <div class="nn-stat">step: <span class="nn-stat-val" id="nn_step">-</span></div>
      <div class="nn-stat">z err: <span class="nn-stat-val" id="nn_zerr">-</span></div>
      <div class="nn-stat">ensemble &mu;: <span class="nn-stat-val" id="nn_emu">-</span></div>
      <div class="nn-stat">ensemble &sigma;: <span class="nn-stat-val" id="nn_esig">-</span></div>
      <div class="bar-wrap">trans err: <span class="bar-bg"><span class="err-bar" id="nn_ebar" style="width:0px;background:#ff9d3b"></span></span></div>
    </div>
  </div>
</div>

<div class="wrap">
  <div class="panel"><canvas id="radar" width="340" height="340"></canvas>
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
      <div class="stat-card color-blue">
        <div class="stat-label">FREE ENERGY</div>
        <div class="stat-value" id="F">-</div>
      </div>
      <div class="stat-card color-orange">
        <div class="stat-label">SENSORY ERR</div>
        <div class="stat-value" id="err">-</div>
      </div>
      <div class="stat-card color-gold">
        <div class="stat-label">EPISTEMIC</div>
        <div class="stat-value" id="epi">-</div>
      </div>
      <div class="stat-card color-gold">
        <div class="stat-label">EPI MAX</div>
        <div class="stat-value" id="epimax">-</div>
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
  </div>
  
  <div class="panel"><canvas id="trace" width="300" height="200"></canvas>
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
  ctx.fillStyle='#0f1218';ctx.fillRect(0,0,W,H);

  const obs=s.obs||[],sAct=s.s||[],pred=s.pred||[],zAbs=s.z_abs||[];
  const eZ=s.e_z_abs||[],eO=s.e_o||[];
  if(!sAct.length)return;

  // Layout: 920x380 canvas, three wide-spaced columns.
  // c1X=observation (left), c2X=latent (center), c3X=prediction (right)
  const PAD=28;
  const TOP=26,BOT=H-PAD-12,ly=TOP,ry=BOT,CH=W-PAD*2;
  const c1X=PAD+CH*0.13;   // observation column x
  const c2X=PAD+CH*0.50;   // latent column x (centered)
  const c3X=PAD+CH*0.87;   // prediction column x
  const midY=(ly+ry)*0.5;

  // ── 1. Column labels at top ────────────────────────────────────────
  ctx.font='bold 11px system-ui';ctx.fillStyle='#6a7888';ctx.textAlign='center';
  ctx.fillText('OBSERVATION  (lidar)',c1X,TOP-6);
  ctx.fillText('LATENT  z  (64 dims)',c2X,TOP-6);
  ctx.fillText('PREDICTION  (what brain expects)',c3X,TOP-6);

  // ── 2. Build node Y positions ──────────────────────────────────────
  const N_DISP=Math.min(24,obs.length,pred.length);
  const obsStep=Math.max(1,Math.floor(obs.length/N_DISP));
  const predStep=Math.max(1,Math.floor(pred.length/N_DISP));
  const obsY=[],predY=[];
  for(let i=0;i<N_DISP;i++){
    const t=(i+0.5)/N_DISP;
    obsY.push(ly+t*(ry-ly));
    predY.push(ly+t*(ry-ly));
  }
  // Latent grid: 8 cols × 8 rows.
  const zNodes=Math.min(sAct.length,zAbs.length);
  const zCols=8,zRows=Math.ceil(zNodes/zCols);
  const zX=[],zY=[];
  const zSpan=CH*0.36;
  for(let r=0;r<zRows;r++){
    for(let c3=0;c3<zCols;c3++){
      const idx=r*zCols+c3;if(idx>=zNodes)break;
      zX.push(c2X+(c3-(zCols-1)*0.5)*(zSpan/Math.max(1,zCols-1)));
      zY.push(ly+(r+1)*(ry-ly)/(zRows+1));
    }
  }

  // ── 3. Connections: latent (center) → prediction (right) ───────────
  // Draw top-K strongest actual activation flows: weight * latent_activation.
  const W_o = s.W_o || [];
  const conns = [];
  let maxFlow = 0.001;
  if (W_o.length && sAct.length) {
    for (let zi = 0; zi < zX.length; zi++) {
      const act = sAct[zi] || 0;
      for (let pi = 0; pi < predY.length; pi++) {
        const bin_idx = Math.min(pi * predStep, W_o.length - 1);
        const weight = W_o[bin_idx][zi] || 0;
        const flow = weight * act;
        const absFlow = Math.abs(flow);
        if (absFlow > maxFlow) {
          maxFlow = absFlow;
        }
        conns.push({ zi, pi, flow, absFlow });
      }
    }
  }
  conns.sort((a, b) => b.absFlow - a.absFlow);
  const connK = Math.min(250, conns.length);
  for (let i = 0; i < connK; i++) {
    const { zi, pi, flow, absFlow } = conns[i];
    const t = absFlow / maxFlow;
    // Positive flow (excitation) = cyan/green; negative flow (inhibition) = orange/red
    const color = flow >= 0 ? `rgba(0,200,140,${0.04+t*0.65})` : `rgba(255,100,50,${0.04+t*0.65})`;
    ctx.beginPath();
    ctx.moveTo(zX[zi] + 6, zY[zi]);
    ctx.lineTo(c3X - 12, predY[pi]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.4 + t * 2.8;
    ctx.stroke();
  }

  // ── 4. Observation nodes (left column) ─────────────────────────────
  const mxObs=Math.max(...obs,0.001);
  obsY.forEach((ny,i)=>{
    const v=obs[Math.min(i*obsStep,obs.length-1)]||0;
    const br=0.2+0.8*(v/mxObs);
    ctx.beginPath();ctx.arc(c1X,ny,4.5,0,2*Math.PI);
    ctx.fillStyle=`rgba(${Math.round(110*br+15)},${Math.round(110*br+15)},${Math.round(110*br+15)},0.88)`;
    ctx.fill();
  });

  // ── 5. Latent nodes (center) — the brain lighting up ───────────────
  const mxZ=Math.max(...zAbs,0.001);
  zX.forEach((nx,zi)=>{
    const ny=zY[zi];
    const absAct=zAbs[zi]||0;
    const activation=sAct[zi]||0;
    const radius=3.5+absAct*11;
    const glowR=radius+absAct*10;
    const[ar,ag,ab]=actColor(activation,mxZ);
    // outer glow (large, transparent halo)
    ctx.beginPath();ctx.arc(nx,ny,glowR,0,2*Math.PI);
    ctx.fillStyle=`rgba(${ar},${ag},${ab},0.08)`;ctx.fill();
    // inner glow ring
    ctx.beginPath();ctx.arc(nx,ny,glowR*0.65,0,2*Math.PI);
    ctx.fillStyle=`rgba(${ar},${ag},${ab},0.16)`;ctx.fill();
    // core with radial gradient for 3-D depth
    const grad=ctx.createRadialGradient(nx-2,ny-2,0,nx,ny,radius);
    grad.addColorStop(0,`rgb(${Math.min(255,ar+80)},${Math.min(255,ag+60)},${Math.min(255,ab+25)})`);
    grad.addColorStop(1,`rgb(${ar},${ag},${ab})`);
    ctx.beginPath();ctx.arc(nx,ny,radius,0,2*Math.PI);
    ctx.fillStyle=grad;ctx.fill();
    // white ring on highly active neurons
    if(absAct>mxZ*0.5){ctx.strokeStyle='rgba(255,255,255,0.8)';ctx.lineWidth=1.1;ctx.stroke();}
  });

  // ── 6. Prediction nodes (right column) ─────────────────────────────
  const mxE2=Math.max(...eO.map(Math.abs),0.001);
  predY.forEach((ny,i)=>{
    const pi=Math.min(i*predStep,pred.length-1);
    const err=Math.abs(eO[pi]||0);
    const[er,eg,eb]=errColor(err,mxE2);
    const bright=Math.min(1,err/mxE2);
    // glow halo
    ctx.beginPath();ctx.arc(c3X,ny,5+10*bright,0,2*Math.PI);
    ctx.fillStyle=`rgba(${er},${eg},${eb},${0.1*bright})`;ctx.fill();
    // core node
    ctx.beginPath();ctx.arc(c3X,ny,4+8*bright,0,2*Math.PI);
    ctx.fillStyle=`rgba(${Math.round(er*bright+28*(1-bright))},${Math.round(eg*bright+28*(1-bright))},${Math.round(eb*bright+28*(1-bright))},0.94)`;
    ctx.fill();
    if(err>mxE2*0.45){ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();}
  });

  // ── 7. Flow arrows between columns ─────────────────────────────────
  const drawArrow=(x,y,dir)=>{
    ctx.fillStyle='#4ab8ff';
    ctx.beginPath();
    if(dir>0){ctx.moveTo(x,y);ctx.lineTo(x-12,y-7);ctx.lineTo(x-12,y+7);}
    else       {ctx.moveTo(x,y);ctx.lineTo(x+12,y-7);ctx.lineTo(x+12,y+7);}
    ctx.closePath();ctx.fill();
  };
  drawArrow(c1X+14,midY,1);
  drawArrow(c3X-14,midY,-1);

  // ── 8. Column divider lines ─────────────────────────────────────────
  ctx.strokeStyle='#25303e';ctx.lineWidth=1.5;
  [c1X+10,c3X-10].forEach(x=>{
    ctx.beginPath();ctx.moveTo(x,ly);ctx.lineTo(x,ry);ctx.stroke();
  });

  // ── 9. Error legend gradient ─────────────────────────────────────────
  const bw=90,bh=7,bx=PAD,by=BOT+8;
  const grd=ctx.createLinearGradient(bx,0,bx+bw,0);
  grd.addColorStop(0,'#1a1208');grd.addColorStop(1,'#ffb840');
  ctx.fillStyle=grd;ctx.fillRect(bx,by,bw,bh);
  ctx.font='10px system-ui';ctx.fillStyle='#4a5565';ctx.textAlign='left';
  ctx.fillText('pred error: dim=correct  bright=wrong',bx,BOT+22);

  // ── 10. State error stats (top-right of canvas) ────────────────────
  const eZnon=eZ.filter(x=>!isNaN(x));
  if(eZnon.length){
    const sumE=eZnon.reduce((a,b)=>a+b,0);
    const avgE=sumE/eZnon.length;
    const mxE3=Math.max(...eZnon);
    ctx.font='11px system-ui';ctx.fillStyle='#5a6678';ctx.textAlign='right';
    ctx.fillText(`z error  avg:${avgE.toFixed(3)}  max:${mxE3.toFixed(3)}`,W-PAD,12);
  }
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
  // Update the dedicated stat row under the neural net canvas.
  const te=s.trans_errors||[];
  if(s.step!=null)$('nn_step').textContent=s.step;
  const eZ=s.e_z_abs||[];
  if(eZ.length){
    const avg=eZ.reduce((a,b)=>a+b,0)/eZ.length;
    $('nn_zerr').textContent=avg.toFixed(4);
  }
  if(te.length){
    const m=te.reduce((a,b)=>a+b,0)/te.length;
    const std=Math.sqrt(te.reduce((a,b)=>a+(b-m)*(b-m),0)/te.length);
    $('nn_emu').textContent=m.toFixed(4);
    $('nn_esig').textContent=std.toFixed(4);
    // Scale bar: max realistic range ~[0,0.1]
    const barW=Math.min(80,Math.round(m/0.1*80));
    $('nn_ebar').style.width=barW+'px';
  }
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
    const badge = $('status_badge');
    const statusText = $('status');
    if (age > 1.5) {
      badge.className = 'status-badge stale';
      statusText.textContent = 'stale (' + age.toFixed(1) + 's)';
    } else {
      badge.className = 'status-badge live';
      statusText.textContent = 'live';
    }
    radar(s);tracks(s);trace(s);brainViz(s);nnStatus(s);
  }catch(e){
    const badge = $('status_badge');
    const statusText = $('status');
    if (badge) badge.className = 'status-badge disconnected';
    if (statusText) statusText.textContent = 'disconnected';
  }
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
                state["W_o"] = [[round(float(val), 3) for val in row] for row in W_o.tolist()]
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
