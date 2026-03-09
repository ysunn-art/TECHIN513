import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// SIMULATED DATA GENERATOR (based on real PsychoMouse patterns)
// ============================================================
function generateSessionData(durationSec = 180) {
  const fps = 30;
  const totalFrames = durationSec * fps;
  const data = [];
  let x = 960, y = 540;
  let vx = 0, vy = 0;

  for (let i = 0; i < totalFrames; i++) {
    const t = i / fps;
    const fatigueFactor = Math.min(1, Math.max(0, (t - 30) / 120));
    const phase = t < 30 ? "Alert" : t < 90 ? "Transition" : "Fatigued";

    // Fatigue effects: slower, more tremor, longer pauses
    const baseSpeed = 8 - fatigueFactor * 4;
    const tremor = fatigueFactor * 2.5;
    const pauseProb = 0.02 + fatigueFactor * 0.06;

    const isPaused = Math.random() < pauseProb;
    if (!isPaused) {
      // Intentional movement with fatigue-dependent smoothness
      if (Math.random() < 0.05) {
        vx = (Math.random() - 0.5) * baseSpeed * 20;
        vy = (Math.random() - 0.5) * baseSpeed * 20;
      }
      vx *= 0.92; vy *= 0.92;
      // Micro-tremor (8-12 Hz band)
      const tremorX = tremor * Math.sin(2 * Math.PI * 10 * t + Math.random());
      const tremorY = tremor * Math.cos(2 * Math.PI * 9.5 * t + Math.random());
      x += vx + tremorX + (Math.random() - 0.5) * 2;
      y += vy + tremorY + (Math.random() - 0.5) * 2;
    }

    x = Math.max(50, Math.min(1870, x));
    y = Math.max(50, Math.min(1030, y));

    const speed = Math.sqrt(vx * vx + vy * vy) + Math.random() * 5;
    const jerk = Math.abs(speed - (data[i - 1]?.speed || 0)) * fps + tremor * 50;

    // Click events
    const clickRate = isPaused ? 0 : (Math.random() < (0.008 - fatigueFactor * 0.004) ? 1 : 0);

    data.push({ t, x, y, speed, jerk, phase, fatigueFactor, tremor, clickRate, isPaused });
  }
  return data;
}

// FFT simulation
function computeFFT(data, currentIdx, windowSize = 90) {
  const start = Math.max(0, currentIdx - windowSize);
  const slice = data.slice(start, currentIdx);
  if (slice.length < 10) return [];

  const avgTremor = slice.reduce((s, d) => s + d.tremor, 0) / slice.length;
  const bins = [];
  for (let f = 0; f <= 50; f += 0.5) {
    let power = 0;
    // Base noise floor
    power += Math.random() * 0.3 + 0.1;
    // Movement energy (0-5 Hz)
    if (f < 5) power += 8 * Math.exp(-f * 0.5) + Math.random() * 2;
    // Tremor band (8-12 Hz) - scales with fatigue
    if (f >= 6 && f <= 14) {
      const tremorPeak = Math.exp(-((f - 10) ** 2) / 4) * avgTremor * 8;
      power += tremorPeak;
    }
    // High frequency noise
    if (f > 20) power += Math.random() * 0.05;
    bins.push({ freq: f, power: Math.max(0, power) });
  }
  return bins;
}

// ============================================================
// COMPONENTS
// ============================================================

const COLORS = {
  bg: "#0a0a0f",
  panel: "#12121a",
  panelBorder: "#1e1e2e",
  accent: "#e8651a",
  accentGlow: "rgba(232,101,26,0.3)",
  alert: "#22c55e",
  alertGlow: "rgba(34,197,94,0.15)",
  transition: "#eab308",
  fatigued: "#ef4444",
  fatiguedGlow: "rgba(239,68,68,0.15)",
  text: "#e2e2e8",
  textDim: "#6b6b80",
  grid: "#1a1a28",
  tremor: "#f59e0b",
};

const phaseColor = (phase) =>
  phase === "Alert" ? COLORS.alert : phase === "Transition" ? COLORS.transition : COLORS.fatigued;

// Mini canvas-based chart component
function CanvasChart({ data, width, height, color, label, unit, gridLines = 4, filled = false, yMin, yMax }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const min = yMin !== undefined ? yMin : Math.min(...data);
    const max = yMax !== undefined ? yMax : Math.max(...data);
    const range = max - min || 1;
    const pad = { top: 8, bottom: 8, left: 0, right: 0 };
    const cw = width - pad.left - pad.right;
    const ch = height - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= gridLines; i++) {
      const gy = pad.top + (ch / gridLines) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(width - pad.right, gy); ctx.stroke();
    }

    // Data line
    ctx.beginPath();
    data.forEach((v, i) => {
      const px = pad.left + (i / (data.length - 1)) * cw;
      const py = pad.top + ch - ((v - min) / range) * ch;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (filled) {
      const lastX = pad.left + cw;
      ctx.lineTo(lastX, pad.top + ch);
      ctx.lineTo(pad.left, pad.top + ch);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, color + "40");
      grad.addColorStop(1, color + "05");
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }, [data, width, height, color, filled, gridLines, yMin, yMax]);

  return (
    <div style={{ position: "relative" }}>
      <canvas ref={canvasRef} style={{ width, height, display: "block" }} />
    </div>
  );
}

// FFT Bar chart
function FFTChart({ bins, width, height }) {
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

    const maxPower = Math.max(...bins.map(b => b.power), 1);
    const barW = width / bins.length;
    const pad = { top: 8, bottom: 20 };
    const ch = height - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const gy = pad.top + (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(width, gy); ctx.stroke();
    }

    // Tremor band highlight (8-12 Hz)
    const tremorStart = bins.findIndex(b => b.freq >= 8);
    const tremorEnd = bins.findIndex(b => b.freq > 12);
    if (tremorStart >= 0 && tremorEnd >= 0) {
      ctx.fillStyle = "rgba(245,158,11,0.08)";
      ctx.fillRect(tremorStart * barW, pad.top, (tremorEnd - tremorStart) * barW, ch);
      ctx.fillStyle = COLORS.tremor + "60";
      ctx.font = "9px monospace";
      ctx.fillText("8-12Hz", tremorStart * barW + 2, pad.top + 12);
    }

    // Bars
    bins.forEach((b, i) => {
      const bh = (b.power / maxPower) * ch;
      const bx = i * barW;
      const by = pad.top + ch - bh;
      const inTremor = b.freq >= 8 && b.freq <= 12;
      ctx.fillStyle = inTremor ? COLORS.tremor : COLORS.accent + "90";
      ctx.fillRect(bx, by, Math.max(1, barW - 1), bh);
    });

    // X-axis labels
    ctx.fillStyle = COLORS.textDim;
    ctx.font = "9px monospace";
    [0, 10, 20, 30, 40, 50].forEach(f => {
      const idx = bins.findIndex(b => b.freq >= f);
      if (idx >= 0) ctx.fillText(f + "Hz", idx * barW, height - 4);
    });
  }, [bins, width, height]);

  return <canvas ref={canvasRef} style={{ width, height, display: "block" }} />;
}

// Mouse trajectory canvas
function TrajectoryCanvas({ points, width, height, phase }) {
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

    // Scale points to canvas
    const scaleX = width / 1920;
    const scaleY = height / 1080;

    // Trail (last 200 points)
    const trail = points.slice(-200);
    trail.forEach((p, i) => {
      if (i === 0) return;
      const alpha = (i / trail.length) * 0.8;
      ctx.beginPath();
      ctx.moveTo(trail[i - 1].x * scaleX, trail[i - 1].y * scaleY);
      ctx.lineTo(p.x * scaleX, p.y * scaleY);
      ctx.strokeStyle = phaseColor(phase) + Math.round(alpha * 255).toString(16).padStart(2, "0");
      ctx.lineWidth = 1 + alpha;
      ctx.stroke();
    });

    // Current position dot
    const cur = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(cur.x * scaleX, cur.y * scaleY, 5, 0, Math.PI * 2);
    ctx.fillStyle = phaseColor(phase);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cur.x * scaleX, cur.y * scaleY, 10, 0, Math.PI * 2);
    ctx.strokeStyle = phaseColor(phase) + "40";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [points, width, height, phase]);

  return <canvas ref={canvasRef} style={{ width, height, display: "block", borderRadius: 8 }} />;
}

// Fatigue Gauge
function FatigueGauge({ value, phase }) {
  const angle = -135 + value * 270;
  const color = phaseColor(phase);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <svg viewBox="0 0 200 130" width="200" height="130">
        {/* Background arc */}
        <path d="M 20 120 A 80 80 0 0 1 180 120" fill="none" stroke={COLORS.grid} strokeWidth="12" strokeLinecap="round" />
        {/* Colored arc */}
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={COLORS.alert} />
            <stop offset="50%" stopColor={COLORS.transition} />
            <stop offset="100%" stopColor={COLORS.fatigued} />
          </linearGradient>
        </defs>
        <path d="M 20 120 A 80 80 0 0 1 180 120" fill="none" stroke="url(#gaugeGrad)" strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${value * 251.2} 251.2`} />
        {/* Needle */}
        <line x1="100" y1="120" x2={100 + 60 * Math.cos((angle * Math.PI) / 180)} y2={120 + 60 * Math.sin((angle * Math.PI) / 180)}
          stroke={color} strokeWidth="3" strokeLinecap="round" />
        <circle cx="100" cy="120" r="6" fill={color} />
        {/* Labels */}
        <text x="15" y="128" fill={COLORS.alert} fontSize="9" fontFamily="monospace">Alert</text>
        <text x="155" y="128" fill={COLORS.fatigued} fontSize="9" fontFamily="monospace">Fatigued</text>
      </svg>
      <div style={{
        fontSize: 28, fontWeight: 800, letterSpacing: -1,
        color, textShadow: `0 0 20px ${color}60`,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {Math.round(value * 100)}%
      </div>
      <div style={{
        fontSize: 11, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase",
        color, padding: "3px 12px", borderRadius: 4,
        background: color + "18", border: `1px solid ${color}40`,
      }}>
        {phase}
      </div>
    </div>
  );
}

// Panel wrapper
function Panel({ title, children, style, span = 1 }) {
  return (
    <div style={{
      background: COLORS.panel, borderRadius: 12,
      border: `1px solid ${COLORS.panelBorder}`,
      padding: "14px 16px", gridColumn: `span ${span}`,
      display: "flex", flexDirection: "column", gap: 8,
      ...style,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 600, letterSpacing: 2.5, textTransform: "uppercase",
        color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace",
      }}>{title}</div>
      {children}
    </div>
  );
}

// Stat display
function Stat({ label, value, unit, color = COLORS.text }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 1, textTransform: "uppercase", fontFamily: "monospace" }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 3 }}>
        <span style={{ fontSize: 22, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>{value}</span>
        {unit && <span style={{ fontSize: 10, color: COLORS.textDim, fontFamily: "monospace" }}>{unit}</span>}
      </div>
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function PsychoMouseDemo() {
  const [sessionData] = useState(() => generateSessionData(180));
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(8);
  const animRef = useRef(null);
  const lastTimeRef = useRef(0);

  const WINDOW = 150; // chart history window

  const tick = useCallback((timestamp) => {
    if (!lastTimeRef.current) lastTimeRef.current = timestamp;
    const dt = timestamp - lastTimeRef.current;
    if (dt > 1000 / 30) {
      lastTimeRef.current = timestamp;
      setFrameIdx(prev => {
        const next = prev + playSpeed;
        if (next >= sessionData.length) { setPlaying(false); return sessionData.length - 1; }
        return next;
      });
    }
    animRef.current = requestAnimationFrame(tick);
  }, [playSpeed, sessionData]);

  useEffect(() => {
    if (playing) {
      animRef.current = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(animRef.current);
    }
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, tick]);

  const cur = sessionData[frameIdx] || sessionData[0];
  const startIdx = Math.max(0, frameIdx - WINDOW);
  const slice = sessionData.slice(startIdx, frameIdx + 1);

  const speeds = slice.map(d => d.speed);
  const jerks = slice.map(d => d.jerk);
  const trajectoryPts = sessionData.slice(Math.max(0, frameIdx - 200), frameIdx + 1);
  const fftBins = computeFFT(sessionData, frameIdx);

  // Compute stats
  const recentWindow = sessionData.slice(Math.max(0, frameIdx - 90), frameIdx + 1);
  const avgSpeed = recentWindow.length ? (recentWindow.reduce((s, d) => s + d.speed, 0) / recentWindow.length).toFixed(1) : "0";
  const avgJerk = recentWindow.length ? (recentWindow.reduce((s, d) => s + d.jerk, 0) / recentWindow.length).toFixed(0) : "0";
  const clicks = recentWindow.filter(d => d.clickRate > 0).length;
  const pauses = recentWindow.filter(d => d.isPaused).length;
  const tremorEnergy = fftBins.filter(b => b.freq >= 8 && b.freq <= 12).reduce((s, b) => s + b.power, 0);
  const totalEnergy = fftBins.reduce((s, b) => s + b.power, 0) || 1;
  const tremorRatio = ((tremorEnergy / totalEnergy) * 100).toFixed(1);

  const timeStr = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div style={{
      width: "100%", minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'JetBrains Mono', -apple-system, monospace",
      padding: 20, boxSizing: "border-box",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, background: `linear-gradient(135deg, ${COLORS.accent}, #ff8c42)`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 800, color: "#fff",
            boxShadow: `0 4px 20px ${COLORS.accentGlow}`,
          }}>P</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: -0.5, fontFamily: "'Space Grotesk', sans-serif" }}>
              PsychoMouse
            </div>
            <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 2, textTransform: "uppercase" }}>
              Fatigue Detection Demo — Session Replay
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: playing ? COLORS.alert : COLORS.textDim,
            boxShadow: playing ? `0 0 8px ${COLORS.alertGlow}` : "none",
          }} />
          <span style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: 1 }}>
            {playing ? "REPLAYING" : "PAUSED"}
          </span>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr 200px",
        gridTemplateRows: "auto auto auto",
        gap: 12,
      }}>
        {/* Row 1: Trajectory + Gauge */}
        <Panel title="Mouse Trajectory" span={2}>
          <TrajectoryCanvas points={trajectoryPts} width={560} height={240} phase={cur.phase} />
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <Stat label="X" value={Math.round(cur.x)} unit="px" />
            <Stat label="Y" value={Math.round(cur.y)} unit="px" />
            <Stat label="Click Events" value={clicks} unit="/3s" color={COLORS.accent} />
            <Stat label="Pauses" value={pauses} unit="/3s" color={COLORS.tremor} />
          </div>
        </Panel>

        <Panel title="Fatigue Level">
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", flex: 1 }}>
            <FatigueGauge value={cur.fatigueFactor} phase={cur.phase} />
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6, width: "100%" }}>
              <Stat label="Tremor Ratio" value={tremorRatio} unit="%" color={COLORS.tremor} />
              <Stat label="Avg Speed" value={avgSpeed} unit="px/s" />
            </div>
          </div>
        </Panel>

        {/* Row 2: Speed + Jerk charts */}
        <Panel title="Velocity (px/s)">
          <CanvasChart data={speeds} width={280} height={120} color={COLORS.accent} filled yMin={0} yMax={80} />
          <Stat label="Current" value={cur.speed.toFixed(1)} unit="px/s" color={COLORS.accent} />
        </Panel>

        <Panel title="Jerk — Movement Smoothness (px/s³)">
          <CanvasChart data={jerks} width={280} height={120} color="#ef4444" filled yMin={0} yMax={400} />
          <Stat label="Current" value={cur.jerk.toFixed(0)} unit="px/s³" color="#ef4444" />
        </Panel>

        <Panel title="Stats">
          <div style={{ display: "flex", flexDirection: "column", gap: 10, flex: 1, justifyContent: "center" }}>
            <Stat label="Time" value={timeStr(cur.t)} color={COLORS.text} />
            <Stat label="Avg Jerk" value={avgJerk} unit="px/s³" />
            <Stat label="Fatigue %" value={Math.round(cur.fatigueFactor * 100)} unit="%" color={phaseColor(cur.phase)} />
          </div>
        </Panel>

        {/* Row 3: FFT + Pipeline */}
        <Panel title="FFT Power Spectrum" span={2}>
          <FFTChart bins={fftBins} width={560} height={130} />
          <div style={{ display: "flex", gap: 16 }}>
            <Stat label="Tremor Band" value={tremorRatio} unit="%" color={COLORS.tremor} />
            <Stat label="Total Power" value={totalEnergy.toFixed(1)} />
            <div style={{
              display: "flex", alignItems: "center", gap: 6, marginLeft: "auto",
              padding: "4px 10px", borderRadius: 6, background: COLORS.tremor + "15",
              border: `1px solid ${COLORS.tremor}30`, fontSize: 9, color: COLORS.tremor, letterSpacing: 1,
            }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: COLORS.tremor + "40" }} />
              TREMOR BAND 8-12 Hz
            </div>
          </div>
        </Panel>

        <Panel title="Pipeline">
          <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 9, flex: 1, justifyContent: "center" }}>
            {["Raw Input", "Butterworth LP", "Derivatives", "FFT Analysis", "ML Classify"].map((step, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "5px 8px", borderRadius: 6,
                background: i <= 4 ? COLORS.accent + "12" : "transparent",
                border: `1px solid ${i <= 4 ? COLORS.accent + "30" : COLORS.panelBorder}`,
              }}>
                <div style={{
                  width: 16, height: 16, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center",
                  background: COLORS.accent, color: "#fff", fontSize: 8, fontWeight: 700,
                }}>{i + 1}</div>
                <span style={{ color: COLORS.text, letterSpacing: 0.5 }}>{step}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* Timeline / Playback Controls */}
      <div style={{
        marginTop: 12, background: COLORS.panel, borderRadius: 12,
        border: `1px solid ${COLORS.panelBorder}`, padding: "12px 16px",
        display: "flex", alignItems: "center", gap: 16,
      }}>
        {/* Play/Pause */}
        <button onClick={() => {
          if (frameIdx >= sessionData.length - 1) setFrameIdx(0);
          setPlaying(!playing);
        }} style={{
          width: 36, height: 36, borderRadius: 8, border: `1px solid ${COLORS.accent}40`,
          background: playing ? COLORS.accent + "20" : "transparent",
          color: COLORS.accent, fontSize: 16, cursor: "pointer", display: "flex",
          alignItems: "center", justifyContent: "center",
        }}>
          {playing ? "⏸" : "▶"}
        </button>

        {/* Time */}
        <div style={{ fontSize: 13, fontWeight: 600, minWidth: 50 }}>
          {timeStr(cur.t)}
        </div>

        {/* Progress bar */}
        <div style={{ flex: 1, position: "relative", height: 24, cursor: "pointer" }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            setFrameIdx(Math.floor(pct * (sessionData.length - 1)));
          }}>
          {/* Track */}
          <div style={{
            position: "absolute", top: 10, left: 0, right: 0, height: 4,
            borderRadius: 2, background: COLORS.grid, overflow: "hidden",
          }}>
            {/* Phase colors */}
            <div style={{ position: "absolute", left: 0, width: "16.7%", height: "100%", background: COLORS.alert + "50" }} />
            <div style={{ position: "absolute", left: "16.7%", width: "33.3%", height: "100%", background: COLORS.transition + "30" }} />
            <div style={{ position: "absolute", left: "50%", width: "50%", height: "100%", background: COLORS.fatigued + "30" }} />
            {/* Progress */}
            <div style={{
              position: "absolute", left: 0, height: "100%", borderRadius: 2,
              width: `${(frameIdx / (sessionData.length - 1)) * 100}%`,
              background: phaseColor(cur.phase),
            }} />
          </div>
          {/* Thumb */}
          <div style={{
            position: "absolute", top: 6,
            left: `${(frameIdx / (sessionData.length - 1)) * 100}%`,
            transform: "translateX(-50%)",
            width: 12, height: 12, borderRadius: "50%",
            background: phaseColor(cur.phase), border: `2px solid ${COLORS.panel}`,
            boxShadow: `0 0 10px ${phaseColor(cur.phase)}60`,
          }} />
          {/* Phase labels */}
          <div style={{ position: "absolute", top: -2, left: "8%", fontSize: 8, color: COLORS.alert, letterSpacing: 1 }}>ALERT</div>
          <div style={{ position: "absolute", top: -2, left: "30%", fontSize: 8, color: COLORS.transition, letterSpacing: 1 }}>TRANSITION</div>
          <div style={{ position: "absolute", top: -2, left: "70%", fontSize: 8, color: COLORS.fatigued, letterSpacing: 1 }}>FATIGUED</div>
        </div>

        {/* Duration */}
        <div style={{ fontSize: 11, color: COLORS.textDim }}>3:00</div>

        {/* Speed control */}
        <div style={{ display: "flex", gap: 4 }}>
          {[1, 4, 8, 16].map(s => (
            <button key={s} onClick={() => setPlaySpeed(s)} style={{
              padding: "4px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600, cursor: "pointer",
              border: `1px solid ${s === playSpeed ? COLORS.accent : COLORS.panelBorder}`,
              background: s === playSpeed ? COLORS.accent + "20" : "transparent",
              color: s === playSpeed ? COLORS.accent : COLORS.textDim,
              fontFamily: "monospace",
            }}>{s}x</button>
          ))}
        </div>
      </div>
    </div>
  );
}
