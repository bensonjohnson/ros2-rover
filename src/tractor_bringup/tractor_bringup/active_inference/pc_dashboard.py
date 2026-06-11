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

import numpy as np


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
  .stat-card.color-red .stat-value { color: #ff5b5b; }
  
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

  /* Supervisor control bar (only shown when served by brain_supervisor) */
  .ctrl-bar {
    display: none;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
  }
  .ctrl-btn {
    font-family: inherit;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 22px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(30, 34, 43, 0.8);
    color: #cdd3da;
    cursor: pointer;
    transition: filter 0.1s ease, transform 0.05s ease;
  }
  .ctrl-btn:hover:not(:disabled) { filter: brightness(1.3); }
  .ctrl-btn:active:not(:disabled) { transform: scale(0.97); }
  .ctrl-btn:disabled { opacity: 0.3; cursor: default; }
  .ctrl-btn.awake  { border-color: rgba(0,200,140,0.4); color: #00c88c; }
  .ctrl-btn.sleep  { border-color: rgba(147,112,219,0.4); color: #da70d6; }
  .ctrl-btn.stop   { border-color: rgba(255,91,91,0.4); color: #ff5b5b; }
  .ctrl-btn.update { border-color: rgba(91,192,255,0.4); color: #5bc0ff; }
  .ctrl-note { font-size: 11px; color: #7c8694; }
</style></head><body>
<h1>PREDICTIVE-CODING ROVER BRAIN</h1>
<div class="sub">observed vs. predicted lidar &mdash; pure epistemic active inference</div>
<div class="ctrl-bar" id="ctrl_bar">
  <button class="ctrl-btn awake" id="btn_awake">Awake</button>
  <button class="ctrl-btn sleep" id="btn_sleep">Sleep</button>
  <button class="ctrl-btn stop" id="btn_stop">Stop</button>
  <button class="ctrl-btn update" id="btn_update">Update</button>
  <span class="ctrl-note" id="ctrl_note"></span>
</div>
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
      <div class="bar-wrap">members: <span id="nn_ebars" style="display:inline-flex;gap:3px;"></span></div>
      <div class="nn-stat" id="nn_dis_wrap" style="display:none">dis &mu;&#8320;: <span class="nn-stat-val" id="nn_dis">-</span></div>
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
      <div class="status-badge stale" id="safety_badge" style="display:none">
        <span class="status-dot"></span>
        <span>SAFETY HOLD</span>
      </div>
      <div class="status-badge live" id="teleop_badge"
           style="display:none;background:rgba(91,192,255,0.1);border-color:rgba(91,192,255,0.25);color:#5bc0ff">
        <span class="status-dot"></span>
        <span>TELEOP</span>
      </div>
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
      <div class="stat-card color-red">
        <div class="stat-label">PRAGMATIC</div>
        <div class="stat-value" id="prag">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">BATTERY VOLTAGE</div>
        <div class="stat-value" id="bat_volt">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">BATTERY %</div>
        <div class="stat-value" id="bat_pct">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">TICK TIME</div>
        <div class="stat-value" id="tick_ms">-</div>
      </div>
      <div class="stat-card color-blue">
        <div class="stat-label">NOVELTY</div>
        <div class="stat-value" id="nov">-</div>
      </div>
      <div class="stat-card color-gold">
        <div class="stat-label">CURIOSITY GATE</div>
        <div class="stat-value" id="epi_gate">-</div>
      </div>
      <div class="stat-card color-blue">
        <div class="stat-label">ROOM</div>
        <div class="stat-value" id="room">-</div>
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
      <span class="dot" style="background:#ff9d3b"></span>sensory err
      <span class="dot" style="background:#ffd043"></span>epistemic</div></div>

  <div class="panel"><canvas id="novtrace" width="300" height="200"></canvas>
    <div class="legend"><span class="dot" style="background:#39ff14"></span>novelty (felt)
      <span class="dot" style="background:#ff9d3b"></span>novelty (predicted)
      <span class="dot" style="background:#5a6370"></span>target</div></div>
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
let Fh=[],Eh=[],EpiH=[];
function trace(s){
  if(s.F_hist){Fh=s.F_hist;Eh=s.err_hist;EpiH=s.epi_hist||[];}
  const c=$('trace'),ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  // Each series is min-max normalized independently (shapes are comparable,
  // magnitudes are not) — so print each series' actual range alongside.
  const plot=(arr,color)=>{if(!arr||!arr.length)return null;
    const mx=Math.max(...arr,1e-6),mn=Math.min(...arr);
    const rng=(mx-mn)||1;ctx.beginPath();
    for(let i=0;i<arr.length;i++){const x=i/(arr.length-1||1)*W;
      const y=H-((arr[i]-mn)/rng)*(H-26)-5;i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();
    return [mn,mx];};
  const rF=plot(Fh,'#5bc0ff'),rE=plot(Eh,'#ff9d3b'),rEpi=plot(EpiH,'#ffd043');
  ctx.font='9px system-ui';ctx.textAlign='left';
  const fmt=r=>r?r[0].toFixed(3)+' – '+r[1].toFixed(3):'';
  ctx.fillStyle='#5bc0ff';ctx.fillText('F '+fmt(rF),4,10);
  ctx.fillStyle='#ff9d3b';ctx.fillText('err '+fmt(rE),4,20);
  ctx.fillStyle='#ffd043';ctx.fillText('epi '+(rEpi?rEpi[0].toFixed(4)+' – '+rEpi[1].toFixed(4):''),104,10);
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
  // The server pre-computes the top-K strongest activation flows
  // (weight * latent_activation) as [zi, pi, flow] — far cheaper than
  // shipping the whole decoder matrix on every poll.
  const flows = s.flows || [];
  let maxFlow = 0.001;
  for (const [, , flow] of flows) {
    const a = Math.abs(flow);
    if (a > maxFlow) maxFlow = a;
  }
  for (const [zi, pi, flow] of flows) {
    if (zi >= zX.length || pi >= predY.length) continue;
    const t = Math.abs(flow) / maxFlow;
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
    // One mini-bar per ensemble member: shows WHICH member diverges, not
    // just that one does.
    const cont=$('nn_ebars');
    if(cont.children.length!==te.length){
      cont.innerHTML='';
      te.forEach(()=>{
        const bg=document.createElement('span');
        bg.className='bar-bg';bg.style.width='16px';
        const fill=document.createElement('span');
        fill.className='err-bar';fill.style.background='#ff9d3b';
        fill.style.height='100%';fill.style.display='block';
        bg.appendChild(fill);cont.appendChild(bg);
      });
    }
    const mxTe=Math.max(...te,1e-9);
    te.forEach((v,i)=>{
      cont.children[i].firstChild.style.width=Math.max(1,Math.round(v/mxTe*16))+'px';
    });
  }
  if(s.disagreement_before!=null){
    $('nn_dis_wrap').style.display='flex';
    $('nn_dis').textContent=s.disagreement_before.toFixed(5);
  }
}

// ---- Interoception trace: felt vs predicted place novelty ------------------
// The curiosity appetite made visible: green = the novelty the brain FEELS
// (its observation channel), orange = what it PREDICTED to feel. The gap
// closing is the brain learning its own novelty dynamics; both hovering
// under the gray target line is the standing "hunger" that drives it to
// seek new rooms.
let lastClears=null, clearFlashUntil=0;
function novtrace(s){
  const c=$('novtrace'),ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0f1218';ctx.fillRect(0,0,W,H);
  const yOf=v=>H-6-v*(H-26);          // fixed [0,1] scale, label band on top
  // gridlines at 0, .5, 1
  ctx.strokeStyle='#1d2530';ctx.lineWidth=1;
  for(const v of [0,0.5,1]){ctx.beginPath();ctx.moveTo(0,yOf(v));ctx.lineTo(W,yOf(v));ctx.stroke();}
  // target preference line
  if(s.novelty_target!=null){
    ctx.strokeStyle='#5a6370';ctx.setLineDash([4,4]);ctx.beginPath();
    ctx.moveTo(0,yOf(s.novelty_target));ctx.lineTo(W,yOf(s.novelty_target));ctx.stroke();
    ctx.setLineDash([]);
  }
  const plot=(arr,color)=>{if(!arr||!arr.length)return;
    ctx.beginPath();
    for(let i=0;i<arr.length;i++){
      const x=i/(arr.length-1||1)*W;
      i?ctx.lineTo(x,yOf(arr[i])):ctx.moveTo(x,yOf(arr[i]));
    }
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();};
  plot(s.nov_pred_hist,'#ff9d3b');
  plot(s.nov_hist,'#39ff14');
  ctx.font='10px system-ui';ctx.textAlign='left';ctx.fillStyle='#4a5565';
  ctx.fillText('interoception: place novelty (brain input)',8,12);
  // flash on memory clear (lift detected)
  if(s.mem_clears!=null){
    if(lastClears!==null && s.mem_clears>lastClears)clearFlashUntil=Date.now()+2500;
    lastClears=s.mem_clears;
  }
  if(Date.now()<clearFlashUntil){
    ctx.fillStyle='rgba(255,208,67,0.92)';ctx.font='bold 13px system-ui';ctx.textAlign='center';
    ctx.fillText('MEMORY CLEARED — rover was moved',W/2,H-14);ctx.textAlign='left';
  }
}

// ---- Supervisor controls (bar appears only when brain_supervisor serves us)
async function sendControl(action){
  if(action==='awake' && !confirm('Start AWAKE mode? The rover will drive autonomously.'))return;
  if(action==='update' && !confirm('Pull latest code and rebuild?'))return;
  try{
    const r=await fetch('/control',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action})});
    const j=await r.json();
    $('ctrl_note').textContent=j.msg||'';
  }catch(e){$('ctrl_note').textContent='control request failed';}
}
$('btn_awake').onclick=()=>sendControl('awake');
$('btn_sleep').onclick=()=>sendControl('sleep');
$('btn_stop').onclick=()=>sendControl('stop');
$('btn_update').onclick=()=>sendControl('update');

function controls(s){
  const sm=s.supervisor_mode;
  if(sm===undefined)return;          // standalone dashboard: no bar
  $('ctrl_bar').style.display='flex';
  const idle=(sm==='idle');
  $('btn_awake').disabled=!idle;
  $('btn_sleep').disabled=!idle;
  $('btn_update').disabled=!idle;
  $('btn_stop').disabled=(sm==='idle'||sm==='updating');
  if(s.supervisor_note)$('ctrl_note').textContent=s.supervisor_note;
}

async function tick(){
  try{
    const s=await (await fetch('/state')).json();
    controls(s);
    const fix=(x,n=3)=>(x==null?'-':x.toFixed(n));
    $('step').textContent=s.step??'-';
    $('F').textContent=fix(s.F);$('err').textContent=fix(s.err);
    $('epi').textContent=fix(s.epi,4);$('epimax').textContent=fix(s.epi_max,4);
    $('prag').textContent=fix(s.prag,4);
    // Novelty card: felt -> predicted interoception. The pair converging is
    // the brain learning the channel; both low = sated (familiar room).
    const novEl=$('nov');
    if(s.novelty!=null){
      novEl.textContent=fix(s.novelty,2)+' → '+fix(s.novelty_pred,2);
    }else{novEl.textContent='-';}
    $('L').textContent=fix(s.L,2);$('R').textContent=fix(s.R,2);

    // Control tick wall time vs budget (green = headroom, red = overrun)
    const tEl=$('tick_ms');
    if(s.tick_ms!=null){
      tEl.textContent=s.tick_ms.toFixed(1)+' ms';
      const b=s.tick_budget_ms;
      if(b){
        tEl.style.color = s.tick_ms>b ? '#ff5b5b' : (s.tick_ms>0.8*b ? '#ff9d3b' : '#00c88c');
      }
    }else{
      tEl.textContent='-';tEl.style.color='';
    }

    // Curiosity gate: how much weight the epistemic term currently carries.
    // ~0 = model confident, driving on pragmatic+novelty (deliberate);
    // 1 = genuine ensemble disagreement, full curiosity (post-dream / new place).
    const gEl=$('epi_gate');
    if(s.epi_gate!=null){
      gEl.textContent=(s.epi_gate*100).toFixed(0)+'%';
      gEl.style.color = s.epi_gate>0.7 ? '#ffd043' : (s.epi_gate>0.3 ? '#ff9d3b' : '#00c88c');
    }else{
      gEl.textContent='-';gEl.style.color='';
    }

    // Room recognition: place novelty (fingerprint distance) and how many
    // distinct-looking places are remembered. Gold = a new room.
    const rEl=$('room');
    if(s.novelty!=null){
      rEl.textContent=(s.novelty*100).toFixed(0)+'% · '+(s.places_n||0)+' seen';
      rEl.style.color = s.novelty>0.7 ? '#ffd043' : (s.novelty>0.4 ? '#ff9d3b' : '#00c88c');
    }else{
      rEl.textContent='-';rEl.style.color='';
    }

    // Safety gate badge: visible whenever the lidar monitor is holding the tracks
    $('safety_badge').style.display = s.safety_hold ? 'inline-flex' : 'none';
    // Teleop badge: a human is driving; the brain is watching and learning
    $('teleop_badge').style.display = s.teleop ? 'inline-flex' : 'none';

    // Battery readings
    const volt = s.battery_voltage;
    const pct = s.battery_percentage;
    $('bat_volt').textContent = volt != null && volt > 0 ? volt.toFixed(2) + ' V' : '-';
    
    const batPctEl = $('bat_pct');
    if (pct != null && pct > 0) {
      batPctEl.textContent = pct.toFixed(1) + '%';
      const pctCard = batPctEl.parentElement;
      if (pct > 50) {
        pctCard.style.borderColor = 'rgba(0, 200, 140, 0.2)';
        batPctEl.style.color = '#00c88c';
      } else if (pct > 25) {
        pctCard.style.borderColor = 'rgba(255, 157, 59, 0.2)';
        batPctEl.style.color = '#ff9d3b';
      } else {
        pctCard.style.borderColor = 'rgba(255, 91, 91, 0.3)';
        batPctEl.style.color = '#ff5b5b';
      }
    } else {
      batPctEl.textContent = '-';
      batPctEl.style.color = '';
      batPctEl.parentElement.style.borderColor = '';
    }
    
    const age=s.age??99;
    const badge = $('status_badge');
    const statusText = $('status');
    if (s.mode) {
      const prog = (s.epoch != null && s.epoch_total != null)
        ? ' ' + s.epoch + '/' + s.epoch_total : '';
      if (s.mode === 'sws') {
        badge.className = 'status-badge live';
        badge.style.background = 'rgba(255, 157, 59, 0.1)';
        badge.style.borderColor = 'rgba(255, 157, 59, 0.25)';
        badge.style.color = '#ff9d3b';
        statusText.textContent = 'SWS Replay' + prog;
      } else if (s.mode === 'rem') {
        badge.className = 'status-badge live';
        badge.style.background = 'rgba(147, 112, 219, 0.1)';
        badge.style.borderColor = 'rgba(147, 112, 219, 0.25)';
        badge.style.color = '#da70d6';
        statusText.textContent = 'REM Dream' + prog;
      } else {
        badge.className = 'status-badge live';
        badge.style.background = ''; badge.style.borderColor = ''; badge.style.color = '';
        statusText.textContent = s.mode;
      }
    } else if (age > 1.5) {
      badge.className = 'status-badge stale';
      badge.style.background = ''; badge.style.borderColor = ''; badge.style.color = '';
      statusText.textContent = 'stale (' + age.toFixed(1) + 's)';
    } else {
      badge.className = 'status-badge live';
      badge.style.background = ''; badge.style.borderColor = ''; badge.style.color = '';
      statusText.textContent = 'live';
    }
    radar(s);tracks(s);trace(s);novtrace(s);brainViz(s);nnStatus(s);
  }catch(e){
    const badge = $('status_badge');
    const statusText = $('status');
    if (badge) {
      badge.className = 'status-badge disconnected';
      badge.style.background = ''; badge.style.borderColor = ''; badge.style.color = '';
    }
    if (statusText) statusText.textContent = 'disconnected';
  }
}
// Self-scheduling poll: the next request starts only after the previous one
// finishes, so slow links can't pile up overlapping fetches.
async function loop(){ await tick(); setTimeout(loop, 100); }
loop();
</script></body></html>
"""


class PCDashboardState:
    """Thread-safe holder for the latest brain state + short trace history."""

    # How many prediction nodes the page displays and how many of the
    # strongest latent->prediction flows we ship (mirrors the client).
    N_DISP = 24
    TOP_FLOWS = 250

    def __init__(self, history: int = 240):
        self._lock = threading.Lock()
        self._state: dict = {}
        self._stamp = 0.0
        self._last_request = 0.0
        self._F = deque(maxlen=history)
        self._err = deque(maxlen=history)
        self._epi = deque(maxlen=history)
        self._nov = deque(maxlen=history)
        self._nov_pred = deque(maxlen=history)

    def active(self, within: float = 5.0) -> bool:
        """True if a browser polled /state recently — producers use this to
        skip the (comparatively expensive) state capture when nobody is
        watching."""
        with self._lock:
            return (time.monotonic() - self._last_request) < within

    def update(self, *, obs, pred, F, err, epi, epi_max, L, R, step,
                s=None, e_o=None, W_o=None,
                trans_errors=None, z_abs=None, e_z_abs=None, prag=None, mode=None,
                battery_voltage=None, battery_percentage=None,
                tick_ms=None, tick_budget_ms=None, safety_hold=None,
                teleop=None,
                epoch=None, epoch_total=None, disagreement_before=None,
                novelty=None, novelty_pred=None, novelty_target=None,
                epi_gate=None, places_n=None, mem_clears=None,
                proprio=None) -> None:
        # Heavy lifting (top-K flow extraction) happens OUTSIDE the lock.
        flows = None
        if W_o is not None and s is not None:
            flows = self._top_flows(np.asarray(W_o), np.asarray(s))

        with self._lock:
            self._F.append(round(float(F), 4))
            self._err.append(round(float(err), 4))
            self._epi.append(round(float(epi), 5))
            state: dict = {
                "obs": [round(float(x), 3) for x in obs],
                "pred": [round(float(x), 3) for x in pred],
                "F": float(F), "err": float(err),
                "epi": float(epi), "epi_max": float(epi_max),
                "prag": float(prag) if prag is not None else 0.0,
                "L": float(L), "R": float(R), "step": int(step),
            }
            if mode is not None:
                state["mode"] = mode
            if epoch is not None:
                state["epoch"] = int(epoch)
            if epoch_total is not None:
                state["epoch_total"] = int(epoch_total)
            if disagreement_before is not None:
                state["disagreement_before"] = float(disagreement_before)
            if battery_voltage is not None:
                state["battery_voltage"] = float(battery_voltage)
            if battery_percentage is not None:
                state["battery_percentage"] = float(battery_percentage)
            if tick_ms is not None:
                state["tick_ms"] = round(float(tick_ms), 2)
            if tick_budget_ms is not None:
                state["tick_budget_ms"] = round(float(tick_budget_ms), 2)
            if safety_hold is not None:
                state["safety_hold"] = bool(safety_hold)
            if teleop is not None:
                state["teleop"] = bool(teleop)
            # Interoceptive novelty channel (felt vs predicted).
            if novelty is not None:
                state["novelty"] = round(float(novelty), 3)
                self._nov.append(round(float(novelty), 4))
            if novelty_pred is not None:
                state["novelty_pred"] = round(float(novelty_pred), 3)
                self._nov_pred.append(round(float(novelty_pred), 4))
            if novelty_target is not None:
                state["novelty_target"] = round(float(novelty_target), 3)
            if epi_gate is not None:
                state["epi_gate"] = round(float(epi_gate), 3)
            if places_n is not None:
                state["places_n"] = int(places_n)
            if mem_clears is not None:
                state["mem_clears"] = int(mem_clears)
            # Proprio channels as fed to the brain: [wl, wr, roll, pitch,
            # yaw, ax, ay, az] each normalized to [0,1] (0.5 = rest).
            if proprio is not None:
                state["proprio"] = [round(float(x), 4) for x in proprio]
            # Neural net activations for the brain visualizer.
            if s is not None:
                state["s"] = [round(float(x), 4) for x in s]
            if flows is not None:
                state["flows"] = flows
            if e_o is not None:
                state["e_o"] = [round(float(x), 4) for x in e_o]
            if z_abs is not None:
                state["z_abs"] = [round(float(x), 4) for x in z_abs]
            if trans_errors is not None:
                state["trans_errors"] = [round(float(x), 4) for x in trans_errors]
            if e_z_abs is not None:
                state["e_z_abs"] = [round(float(x), 4) for x in e_z_abs]
            self._state = state
            self._stamp = time.monotonic()

    @classmethod
    def _top_flows(cls, W_o: np.ndarray, s: np.ndarray) -> list:
        """Strongest latent->prediction activation flows, as [zi, pi, flow].

        The page draws only N_DISP subsampled prediction nodes, so instead of
        shipping the full decoder matrix (O*D floats per poll) we extract the
        top-K |weight * activation| entries over those rows server-side.
        """
        n_disp = min(cls.N_DISP, W_o.shape[0])
        row_step = max(1, W_o.shape[0] // n_disp)
        rows = W_o[::row_step][:n_disp]              # [n_disp, D]
        flow = rows * s[None, :]                     # [n_disp, D]
        flat = flow.ravel()
        k = min(cls.TOP_FLOWS, flat.size)
        idx = np.argpartition(np.abs(flat), flat.size - k)[flat.size - k:]
        D = flow.shape[1]
        return [[int(i % D), int(i // D), round(float(flat[i]), 4)] for i in idx]

    def snapshot(self) -> dict:
        with self._lock:
            self._last_request = time.monotonic()
            s = dict(self._state)
            if s:
                s["age"] = time.monotonic() - self._stamp
                s["F_hist"] = list(self._F)
                s["err_hist"] = list(self._err)
                s["epi_hist"] = list(self._epi)
                s["nov_hist"] = list(self._nov)
                s["nov_pred_hist"] = list(self._nov_pred)
            return s


def _make_handler(state: PCDashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body = json.dumps(state.snapshot()).encode()
                ctype = "application/json"
            elif self.path in ("/", "/index.html"):
                body = _PAGE.encode()
                ctype = "text/html"
            else:
                # e.g. /favicon.ico — don't answer every path with the page
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
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
