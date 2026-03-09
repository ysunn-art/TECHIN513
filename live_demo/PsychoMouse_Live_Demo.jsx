import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// REAL-TIME SIGNAL PROCESSING ENGINE
// ============================================================

// Butterworth filter approximation (simple IIR low-pass)
class LowPassFilter {
  constructor(cutoff = 0.15) {
    this.alpha = cutoff;
    this.prev = null;
  }
  apply(val) {
    if (this.prev === null) { this.prev = val; return val; }
    this.prev = this.prev + this.alpha * (val - this.prev);
    return this.prev;
  }
  reset() { this.prev = null; }
}

// Circular buffer for efficient windowed stats
class CircularBuffer {
  constructor(size) {
    this.size = size;
    this.data = [];
    this.idx = 0;
  }
  push(val) {
    if (this.data.length < this.size) this.data.push(val);
    else { this.data[this.idx] = val; this.idx = (this.idx + 1) % this.size; }
  }
  toArray() {
    if (this.data.length < this.size) return [...this.data];
    return [...this.data.slice(this.idx), ...this.data.slice(0, this.idx)];
  }
  get length() { return this.data.length; }
  last() { return this.data.length ? this.data[(this.idx - 1 + this.data.length) % this.data.length] : null; }
}

// Simple FFT (radix-2 DIT)
function simpleFFT(signal) {
  const n = signal.length;
  if (n <= 1) return signal.map(v => ({ re: v, im: 0 }));
  
  // Pad to power of 2
  let size = 1;
  while (size < n) size *= 2;
  const padded = new Array(size).fill(0);
  for (let i = 0; i < n; i++) padded[i] = signal[i];
  
  // Bit-reversal permutation
  const X = padded.map(v => ({ re: v, im: 0 }));
  for (let i = 1, j = 0; i < size; i++) {
    let bit = size >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) { const tmp = X[i]; X[i] = X[j]; X[j] = tmp; }
  }
  
  // FFT butterfly
  for (let len = 2; len <= size; len *= 2) {
    const ang = -2 * Math.PI / len;
    const wn = { re: Math.cos(ang), im: Math.sin(ang) };
    for (let i = 0; i < size; i += len) {
      let w = { re: 1, im: 0 };
      for (let j = 0; j < len / 2; j++) {
        const u = X[i + j];
        const v = { re: w.re * X[i + j + len / 2].re - w.im * X[i + j + len / 2].im, im: w.re * X[i + j + len / 2].im + w.im * X[i + j + len / 2].re };
        X[i + j] = { re: u.re + v.re, im: u.im + v.im };
        X[i + j + len / 2] = { re: u.re - v.re, im: u.im - v.im };
        const newW = { re: w.re * wn.re - w.im * wn.im, im: w.re * wn.im + w.im * wn.re };
        w = newW;
      }
    }
  }
  return X;
}

function computePowerSpectrum(signal, sampleRate) {
  if (signal.length < 8) return [];
  // Apply Hanning window
  const windowed = signal.map((v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (signal.length - 1))));
  const fftResult = simpleFFT(windowed);
  const n = fftResult.length;
  const bins = [];
  for (let i = 0; i < n / 2; i++) {
    const freq = (i * sampleRate) / n;
    if (freq > 50) break;
    const power = Math.sqrt(fftResult[i].re ** 2 + fftResult[i].im ** 2) / n;
    bins.push({ freq, power });
  }
  return bins;
}

// ============================================================
// COLORS & CONSTANTS
// ============================================================
const C = {
  bg: "#060610",
  panel: "#0d0d1a",
  panelBorder: "#1a1a30",
  accent: "#e8651a",
  accentLight: "#ff924c",
  alert: "#10b981",
  alertBg: "rgba(16,185,129,0.08)",
  caution: "#f59e0b",
  cautionBg: "rgba(245,158,11,0.08)",
  fatigued: "#ef4444",
  fatiguedBg: "rgba(239,68,68,0.08)",
  text: "#e8e8f0",
  textDim: "#555570",
  grid: "#151528",
  tremor: "#f59e0b",
};

const SAMPLE_RATE = 60; // ~60 fps capture
const CHART_HISTORY = 300; // 5 seconds of data
const FFT_WINDOW = 256;
const FATIGUE_WINDOW = 180; // 3 sec for stats

// ============================================================
// CANVAS COMPONENTS
// ============================================================

function WaveformChart({ data, width, height, color, yMin = 0, yMax = "auto", label, unit, currentVal }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (!data.length) return;
    const min = yMin;
    const max = yMax === "auto" ? Math.max(...data, 1) : yMax;
    const range = max - min || 1;
    const p = { t: 6, b: 6, l: 0, r: 0 };
    const cw = width - p.l - p.r;
    const ch = height - p.t - p.b;

    // Scanline effect
    ctx.fillStyle = color + "06";
    for (let i = 0; i < ch; i += 3) {
      ctx.fillRect(p.l, p.t + i, cw, 1);
    }

    // Grid lines
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const gy = p.t + (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(p.l, gy); ctx.lineTo(width - p.r, gy); ctx.stroke();
    }

    // Waveform
    ctx.beginPath();
    data.forEach((v, i) => {
      const px = p.l + (i / Math.max(data.length - 1, 1)) * cw;
      const py = p.t + ch - ((Math.min(v, max) - min) / range) * ch;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Fill gradient
    const lastPx = p.l + cw;
    ctx.lineTo(lastPx, p.t + ch);
    ctx.lineTo(p.l, p.t + ch);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, color + "25");
    grad.addColorStop(1, color + "02");
    ctx.fillStyle = grad;
    ctx.fill();
  }, [data, width, height, color, yMin, yMax]);

  return (
    <div style={{ position: "relative" }}>
      <canvas ref={canvasRef} style={{ width, height, display: "block" }} />
      {currentVal !== undefined && (
        <div style={{
          position: "absolute", top: 4, right: 6, fontSize: 18, fontWeight: 700,
          color, fontFamily: "'JetBrains Mono', monospace",
          textShadow: `0 0 15px ${color}60`,
        }}>
          {currentVal}<span style={{ fontSize: 9, color: C.textDim, marginLeft: 3 }}>{unit}</span>
        </div>
      )}
    </div>
  );
}

function FFTDisplay({ bins, width, height }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bins.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const maxP = Math.max(...bins.map(b => b.power), 0.01);
    const barW = width / bins.length;
    const p = { t: 6, b: 18 };
    const ch = height - p.t - p.b;

    // Tremor band bg
    const ts = bins.findIndex(b => b.freq >= 8);
    const te = bins.findIndex(b => b.freq > 12);
    if (ts >= 0 && te >= 0) {
      ctx.fillStyle = C.tremor + "0a";
      ctx.fillRect(ts * barW, p.t, (te - ts) * barW, ch);
      // Border lines
      ctx.strokeStyle = C.tremor + "30";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ts * barW, p.t); ctx.lineTo(ts * barW, p.t + ch); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(te * barW, p.t); ctx.lineTo(te * barW, p.t + ch); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Bars
    bins.forEach((b, i) => {
      const bh = (b.power / maxP) * ch;
      const by = p.t + ch - bh;
      const inTremor = b.freq >= 8 && b.freq <= 12;
      
      if (inTremor) {
        const grad = ctx.createLinearGradient(0, by, 0, p.t + ch);
        grad.addColorStop(0, C.tremor);
        grad.addColorStop(1, C.tremor + "40");
        ctx.fillStyle = grad;
      } else {
        ctx.fillStyle = C.accent + "70";
      }
      ctx.fillRect(i * barW, by, Math.max(1, barW - 0.5), bh);
    });

    // X labels
    ctx.fillStyle = C.textDim;
    ctx.font = "9px 'JetBrains Mono', monospace";
    [0, 10, 20, 30, 40, 50].forEach(f => {
      const idx = bins.findIndex(b => b.freq >= f);
      if (idx >= 0) ctx.fillText(f + "", idx * barW, height - 4);
    });
  }, [bins, width, height]);

  return <canvas ref={canvasRef} style={{ width, height, display: "block" }} />;
}

function TrajectoryView({ points, width, height }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length < 2) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Find bounds
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    let minX = Math.min(...xs), maxX = Math.max(...xs);
    let minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 100;
    const rangeY = maxY - minY || 100;
    const pad = 20;
    const scaleX = (width - 2 * pad) / rangeX;
    const scaleY = (height - 2 * pad) / rangeY;
    const scale = Math.min(scaleX, scaleY);
    const offX = pad + (width - 2 * pad - rangeX * scale) / 2;
    const offY = pad + (height - 2 * pad - rangeY * scale) / 2;

    const toScreen = (p) => ({ x: offX + (p.x - minX) * scale, y: offY + (p.y - minY) * scale });

    // Draw trail
    const trail = points.slice(-300);
    for (let i = 1; i < trail.length; i++) {
      const alpha = i / trail.length;
      const a = toScreen(trail[i - 1]);
      const b = toScreen(trail[i]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = C.accent + Math.round(alpha * 200).toString(16).padStart(2, "0");
      ctx.lineWidth = 0.8 + alpha * 1.5;
      ctx.stroke();
    }

    // Current dot with glow
    const cur = toScreen(points[points.length - 1]);
    // Outer glow
    const grad = ctx.createRadialGradient(cur.x, cur.y, 0, cur.x, cur.y, 20);
    grad.addColorStop(0, C.accent + "50");
    grad.addColorStop(1, C.accent + "00");
    ctx.fillStyle = grad;
    ctx.fillRect(cur.x - 20, cur.y - 20, 40, 40);
    // Dot
    ctx.beginPath();
    ctx.arc(cur.x, cur.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = C.accentLight;
    ctx.shadowColor = C.accent;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
    // Ring
    ctx.beginPath();
    ctx.arc(cur.x, cur.y, 10, 0, Math.PI * 2);
    ctx.strokeStyle = C.accent + "50";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [points, width, height]);

  return <canvas ref={canvasRef} style={{ width, height, display: "block", borderRadius: 8 }} />;
}

// ============================================================
// GAUGE COMPONENT
// ============================================================
function FatigueGauge({ score }) {
  const level = score < 30 ? "alert" : score < 65 ? "caution" : "fatigued";
  const color = level === "alert" ? C.alert : level === "caution" ? C.caution : C.fatigued;
  const bgColor = level === "alert" ? C.alertBg : level === "caution" ? C.cautionBg : C.fatiguedBg;
  const labelText = level === "alert" ? "ALERT" : level === "caution" ? "MODERATE" : "FATIGUED";
  const pct = score / 100;
  const angle = -135 + pct * 270;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <svg viewBox="0 0 180 110" width="180" height="110">
        <defs>
          <linearGradient id="liveGaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={C.alert} />
            <stop offset="50%" stopColor={C.caution} />
            <stop offset="100%" stopColor={C.fatigued} />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <path d="M 15 100 A 75 75 0 0 1 165 100" fill="none" stroke={C.grid} strokeWidth="10" strokeLinecap="round" />
        <path d="M 15 100 A 75 75 0 0 1 165 100" fill="none" stroke="url(#liveGaugeGrad)" strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${pct * 235.6} 235.6`} />
        <line x1="90" y1="100"
          x2={90 + 55 * Math.cos((angle * Math.PI) / 180)}
          y2={100 + 55 * Math.sin((angle * Math.PI) / 180)}
          stroke={color} strokeWidth="2.5" strokeLinecap="round" filter="url(#glow)" />
        <circle cx="90" cy="100" r="5" fill={color} filter="url(#glow)" />
      </svg>
      <div style={{
        fontSize: 32, fontWeight: 800, color, letterSpacing: -2,
        fontFamily: "'JetBrains Mono', monospace",
        textShadow: `0 0 25px ${color}50`,
        marginTop: -8,
      }}>
        {Math.round(score)}
      </div>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: 3,
        color, padding: "4px 14px", borderRadius: 6,
        background: bgColor, border: `1px solid ${color}30`,
      }}>
        {labelText}
      </div>
    </div>
  );
}

// ============================================================
// PANEL
// ============================================================
function Panel({ title, children, style, glow }) {
  return (
    <div style={{
      background: C.panel, borderRadius: 14,
      border: `1px solid ${glow ? glow + "30" : C.panelBorder}`,
      padding: "12px 14px", display: "flex", flexDirection: "column", gap: 6,
      boxShadow: glow ? `0 0 30px ${glow}10, inset 0 1px 0 ${glow}10` : "none",
      ...style,
    }}>
      <div style={{
        fontSize: 9, fontWeight: 700, letterSpacing: 3, textTransform: "uppercase",
        color: glow || C.textDim, fontFamily: "'JetBrains Mono', monospace",
      }}>{title}</div>
      {children}
    </div>
  );
}

function MiniStat({ label, value, unit, color = C.text, small }) {
  return (
    <div>
      <div style={{ fontSize: 8, color: C.textDim, letterSpacing: 1.5, textTransform: "uppercase", fontFamily: "monospace", marginBottom: 2 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 2 }}>
        <span style={{ fontSize: small ? 14 : 20, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>{value}</span>
        {unit && <span style={{ fontSize: 9, color: C.textDim }}>{unit}</span>}
      </div>
    </div>
  );
}

// ============================================================
// PIPELINE STEP INDICATOR
// ============================================================
function PipelineBar({ activeData }) {
  const steps = [
    { label: "CAPTURE", icon: "🖱️", desc: "Raw mouse input" },
    { label: "FILTER", icon: "〰️", desc: "Butterworth LP 20Hz" },
    { label: "DERIVE", icon: "∂", desc: "v(t), a(t), j(t)" },
    { label: "FFT", icon: "📊", desc: "Spectral analysis" },
    { label: "CLASSIFY", icon: "🧠", desc: "Fatigue detection" },
  ];

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 2,
      background: C.panel, borderRadius: 10, padding: "8px 12px",
      border: `1px solid ${C.panelBorder}`,
    }}>
      {steps.map((s, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 2 }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "5px 10px", borderRadius: 8,
            background: activeData ? C.accent + "12" : "transparent",
            border: `1px solid ${activeData ? C.accent + "25" : C.panelBorder}`,
            transition: "all 0.3s",
          }}>
            <span style={{ fontSize: 12 }}>{s.icon}</span>
            <div>
              <div style={{ fontSize: 9, fontWeight: 700, color: C.text, letterSpacing: 1, fontFamily: "monospace" }}>{s.label}</div>
              <div style={{ fontSize: 7, color: C.textDim, letterSpacing: 0.5 }}>{s.desc}</div>
            </div>
          </div>
          {i < steps.length - 1 && (
            <div style={{ color: activeData ? C.accent : C.textDim, fontSize: 10, padding: "0 2px" }}>→</div>
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// MAIN LIVE DEMO APP
// ============================================================
export default function PsychoMouseLive() {
  const containerRef = useRef(null);

  // Data buffers
  const posBufferRef = useRef(new CircularBuffer(600));
  const speedBufferRef = useRef(new CircularBuffer(CHART_HISTORY));
  const accelBufferRef = useRef(new CircularBuffer(CHART_HISTORY));
  const jerkBufferRef = useRef(new CircularBuffer(CHART_HISTORY));
  const filtXRef = useRef(new CircularBuffer(FFT_WINDOW));
  const filtYRef = useRef(new CircularBuffer(FFT_WINDOW));
  const filterX = useRef(new LowPassFilter(0.15));
  const filterY = useRef(new LowPassFilter(0.15));

  const prevSpeed = useRef(0);
  const prevAccel = useRef(0);
  const lastMouse = useRef({ x: 0, y: 0, t: Date.now() });
  const mouseActive = useRef(false);
  const clickCount = useRef(0);
  const pauseFrames = useRef(0);
  const totalFrames = useRef(0);
  const startTime = useRef(Date.now());

  // State for rendering
  const [displayState, setDisplayState] = useState({
    speeds: [], accels: [], jerks: [], positions: [],
    fftBins: [], speed: 0, accel: 0, jerk: 0,
    fatigueScore: 0, tremorRatio: 0, elapsed: 0,
    avgSpeed: 0, avgJerk: 0, clicks: 0, pauseRate: 0,
    active: false,
  });

  // Mouse event handler
  const handleMouse = useCallback((e) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    lastMouse.current = { x, y, t: Date.now() };
    mouseActive.current = true;
  }, []);

  const handleClick = useCallback(() => {
    clickCount.current++;
  }, []);

  // Main processing loop
  useEffect(() => {
    let rafId;
    let lastTick = 0;

    const processFrame = (timestamp) => {
      rafId = requestAnimationFrame(processFrame);

      if (timestamp - lastTick < 1000 / SAMPLE_RATE) return;
      lastTick = timestamp;

      const mouse = lastMouse.current;
      const posBuf = posBufferRef.current;
      const speedBuf = speedBufferRef.current;
      const accelBuf = accelBufferRef.current;
      const jerkBuf = jerkBufferRef.current;

      // Raw position
      posBuf.push({ x: mouse.x, y: mouse.y });

      // Filtered position
      const fx = filterX.current.apply(mouse.x);
      const fy = filterY.current.apply(mouse.y);
      filtXRef.current.push(fx);
      filtYRef.current.push(fy);

      // Velocity
      const prevPos = posBuf.length > 1 ? posBuf.toArray().slice(-2)[0] : { x: fx, y: fy };
      const dx = fx - (filterX.current.prev || fx);
      const dy = fy - (filterY.current.prev || fy);
      const speed = Math.sqrt(dx * dx + dy * dy) * SAMPLE_RATE;
      speedBuf.push(speed);

      // Acceleration
      const accel = Math.abs(speed - prevSpeed.current) * SAMPLE_RATE;
      accelBuf.push(accel);
      prevSpeed.current = speed;

      // Jerk
      const jerk = Math.abs(accel - prevAccel.current) * SAMPLE_RATE;
      jerkBuf.push(Math.min(jerk, 50000));
      prevAccel.current = accel;

      // Pause detection
      totalFrames.current++;
      if (speed < 3) pauseFrames.current++;

      // Detect if mouse is not active
      const timeSinceMove = Date.now() - mouse.t;
      const isActive = timeSinceMove < 2000;

      // FFT on filtered X signal (idle segments)
      const filtXArr = filtXRef.current.toArray();
      const fftBins = filtXArr.length >= 32 ? computePowerSpectrum(filtXArr, SAMPLE_RATE) : [];

      // Tremor energy
      const tremorPower = fftBins.filter(b => b.freq >= 8 && b.freq <= 12).reduce((s, b) => s + b.power, 0);
      const totalPower = fftBins.reduce((s, b) => s + b.power, 0) || 1;
      const tremorRatio = tremorPower / totalPower;

      // Fatigue score heuristic (composite of multiple indicators)
      const recentSpeeds = speedBuf.toArray().slice(-FATIGUE_WINDOW);
      const recentJerks = jerkBuf.toArray().slice(-FATIGUE_WINDOW);
      const avgSpd = recentSpeeds.length ? recentSpeeds.reduce((a, b) => a + b, 0) / recentSpeeds.length : 0;
      const avgJrk = recentJerks.length ? recentJerks.reduce((a, b) => a + b, 0) / recentJerks.length : 0;
      const pauseRate = totalFrames.current > 0 ? pauseFrames.current / totalFrames.current : 0;

      // Composite fatigue score (0-100)
      const speedScore = Math.min(1, Math.max(0, 1 - avgSpd / 300)) * 25;
      const tremorScore = Math.min(1, tremorRatio * 15) * 30;
      const pauseScore = Math.min(1, pauseRate * 2) * 25;
      const jerkScore = Math.min(1, avgJrk / 5000) * 20;
      const rawFatigue = speedScore + tremorScore + pauseScore + jerkScore;

      const elapsed = (Date.now() - startTime.current) / 1000;

      // Update state at ~15fps for rendering
      if (totalFrames.current % 4 === 0) {
        setDisplayState({
          speeds: speedBuf.toArray(),
          accels: accelBuf.toArray(),
          jerks: jerkBuf.toArray(),
          positions: posBuf.toArray(),
          fftBins,
          speed: speed.toFixed(1),
          accel: accel.toFixed(0),
          jerk: Math.min(jerk, 50000).toFixed(0),
          fatigueScore: Math.min(100, rawFatigue),
          tremorRatio: (tremorRatio * 100).toFixed(1),
          elapsed,
          avgSpeed: avgSpd.toFixed(1),
          avgJerk: avgJrk.toFixed(0),
          clicks: clickCount.current,
          pauseRate: (pauseRate * 100).toFixed(0),
          active: isActive,
        });
      }
    };

    rafId = requestAnimationFrame(processFrame);
    return () => cancelAnimationFrame(rafId);
  }, []);

  const d = displayState;
  const timeStr = `${Math.floor(d.elapsed / 60)}:${String(Math.floor(d.elapsed % 60)).padStart(2, "0")}`;

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouse}
      onClick={handleClick}
      style={{
        width: "100%", minHeight: "100vh", background: C.bg, color: C.text,
        fontFamily: "'JetBrains Mono', -apple-system, monospace",
        padding: 16, boxSizing: "border-box", cursor: "crosshair",
        userSelect: "none",
      }}
    >
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: `linear-gradient(135deg, ${C.accent}, ${C.accentLight})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 800, color: "#fff",
            boxShadow: `0 2px 15px ${C.accent}40`,
          }}>P</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: -0.3, fontFamily: "'Instrument Sans', sans-serif" }}>
              PsychoMouse <span style={{ color: C.accent }}>Live</span>
            </div>
            <div style={{ fontSize: 8, color: C.textDim, letterSpacing: 2.5 }}>
              REAL-TIME FATIGUE DETECTION — MOVE YOUR MOUSE
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: d.active ? C.alert : C.textDim,
              boxShadow: d.active ? `0 0 8px ${C.alert}60` : "none",
              animation: d.active ? "pulse 1.5s infinite" : "none",
            }} />
            <span style={{ fontSize: 9, color: d.active ? C.alert : C.textDim, letterSpacing: 1.5, fontWeight: 600 }}>
              {d.active ? "TRACKING" : "IDLE"}
            </span>
          </div>
          <div style={{ fontSize: 16, fontWeight: 600, color: C.text, fontFamily: "'JetBrains Mono', monospace" }}>
            {timeStr}
          </div>
        </div>
      </div>

      {/* Pipeline Bar */}
      <PipelineBar activeData={d.active} />

      {/* Main Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1.2fr 1fr 200px",
        gridTemplateRows: "auto auto auto",
        gap: 10, marginTop: 10,
      }}>
        {/* Trajectory */}
        <Panel title="Cursor Trajectory" style={{ gridRow: "span 2" }}>
          <TrajectoryView points={d.positions} width={340} height={280} />
          <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
            <MiniStat label="Clicks" value={d.clicks} small />
            <MiniStat label="Pause %" value={d.pauseRate} unit="%" small color={C.caution} />
          </div>
        </Panel>

        {/* Speed */}
        <Panel title="Velocity">
          <WaveformChart data={d.speeds} width={290} height={100} color={C.accent}
            yMin={0} yMax={500} currentVal={d.speed} unit="px/s" />
        </Panel>

        {/* Gauge */}
        <Panel title="Fatigue Index" glow={d.fatigueScore > 60 ? C.fatigued : d.fatigueScore > 30 ? C.caution : undefined}
          style={{ gridRow: "span 2", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <FatigueGauge score={d.fatigueScore} />
          <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%", marginTop: 8 }}>
            <MiniStat label="Tremor 8-12Hz" value={d.tremorRatio} unit="%" color={C.tremor} small />
            <MiniStat label="Avg Speed" value={d.avgSpeed} unit="px/s" small />
            <MiniStat label="Avg Jerk" value={d.avgJerk} unit="px/s³" small />
          </div>
        </Panel>

        {/* Jerk */}
        <Panel title="Jerk — Smoothness">
          <WaveformChart data={d.jerks} width={290} height={100} color="#ef4444"
            yMin={0} yMax={20000} currentVal={d.jerk} unit="px/s³" />
        </Panel>

        {/* FFT */}
        <Panel title="FFT Power Spectrum" style={{ gridColumn: "span 2" }}>
          <FFTDisplay bins={d.fftBins} width={650} height={130} />
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <MiniStat label="Tremor Ratio" value={d.tremorRatio} unit="%" color={C.tremor} small />
            <div style={{
              display: "flex", alignItems: "center", gap: 5, marginLeft: "auto",
              padding: "3px 10px", borderRadius: 5, background: C.tremor + "10",
              border: `1px solid ${C.tremor}25`, fontSize: 8, color: C.tremor, letterSpacing: 1.5, fontWeight: 600,
            }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: C.tremor + "50" }} />
              TREMOR BAND 8–12 Hz
            </div>
            <div style={{ fontSize: 8, color: C.textDim, letterSpacing: 1 }}>Hz</div>
          </div>
        </Panel>
      </div>

      {/* Instruction overlay (fades after first move) */}
      {!d.active && d.elapsed < 5 && (
        <div style={{
          position: "fixed", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(6,6,16,0.7)", zIndex: 100, backdropFilter: "blur(4px)",
        }}>
          <div style={{
            textAlign: "center", padding: "40px 60px", borderRadius: 20,
            background: C.panel, border: `1px solid ${C.panelBorder}`,
            boxShadow: `0 0 60px ${C.accent}15`,
          }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🖱️</div>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Instrument Sans', sans-serif", marginBottom: 8 }}>
              Move your mouse to begin
            </div>
            <div style={{ fontSize: 11, color: C.textDim, letterSpacing: 1 }}>
              Real-time signal processing starts automatically
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
