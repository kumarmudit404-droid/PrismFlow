/* Motion: the ambient background and the 3D prism.
 *
 * Presentation only. Nothing here reads, writes or animates a data value --
 * numbers are in the DOM at their committed values before any animation
 * starts and are never tweened.
 *
 * The 3D prism is optional. three.js is imported dynamically, only once WebGL
 * 2 is actually available and motion is not reduced. If the import fails, the
 * context is lost, or the module throws, the static SVG mark in #prism-bg
 * stays where it is and the page is unchanged.
 *
 * Every colour is read from tokens.css at runtime, so the palette cannot
 * drift from the approved tokens. No hex literal appears in this file.
 */

import { animate, stagger } from "../vendor/anime/anime.esm.min.js";

const root = document.documentElement;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

/* A missing token returns null rather than a guessed fallback, and the caller
 * skips that beam -- inventing a colour here would be the palette drift this
 * indirection exists to prevent. */
function token(name) {
  const v = getComputedStyle(root).getPropertyValue(name).trim();
  return v || null;
}

const ANGLES = [
  { key: "regulatory", token: "--angle-regulatory", unbuilt: true },
  { key: "sentiment", token: "--angle-sentiment" },
  { key: "tech", token: "--angle-tech" },
  { key: "market", token: "--angle-market" },
  { key: "financial", token: "--angle-financial" },
];

function palette() {
  return ANGLES.map((a) => ({ ...a, colour: token(a.token) })).filter((a) => a.colour);
}

/* ====================================================================
 * 2. Ambient background -- ONE canvas, never a stack of blurred layers
 * ==================================================================== */

class Ambient {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.blobs = [];
    this.dpr = 1;
    this.quality = 1;      // dropped to 0 if the frame budget is missed
    this.resize();
    window.addEventListener("resize", () => this.resize(), { passive: true });
  }

  resize() {
    // DPR is capped: a 3x display would otherwise quadruple the fill cost of
    // full-viewport radial gradients for no visible gain on a soft blob.
    this.dpr = Math.min(window.devicePixelRatio || 1, this.quality ? 1.5 : 1);
    const w = window.innerWidth, h = window.innerHeight;
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this.w = w; this.h = h;
    if (!this.blobs.length) { this.seed(); }
  }

  seed() {
    const pal = palette();
    const count = this.quality ? pal.length : 3;
    this.blobs = [];
    for (let i = 0; i < count; i++) {
      const a = pal[i % pal.length];
      this.blobs.push({
        colour: a.colour,
        unbuilt: !!a.unbuilt,
        x: Math.random(), y: Math.random(),
        r: 0.28 + Math.random() * 0.22,
        // slow: a background that reads as motion is a background that
        // competes with the text in front of it
        vx: (Math.random() - 0.5) * 0.000045,
        vy: (Math.random() - 0.5) * 0.000035,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }

  frame(t) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.globalCompositeOperation = "lighter";

    for (const b of this.blobs) {
      b.x += b.vx * 16; b.y += b.vy * 16;
      if (b.x < -0.3) b.x = 1.3; if (b.x > 1.3) b.x = -0.3;
      if (b.y < -0.3) b.y = 1.3; if (b.y > 1.3) b.y = -0.3;

      const cx = b.x * this.w;
      const cy = b.y * this.h + Math.sin(t / 5200 + b.phase) * 14;
      const rad = b.r * Math.max(this.w, this.h);

      // Regulatory is the unbuilt angle and must not look alive anywhere,
      // including here.
      //
      // These two alphas are solved against the contrast budget, not chosen
      // by eye: with additive compositing all five blobs can stack on one
      // pixel, and higher values pushed body text below its contrast
      // threshold. See web/README.md.
      const alpha = b.unbuilt ? 0.023 : 0.05;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
      g.addColorStop(0, hexA(b.colour, alpha));
      g.addColorStop(1, hexA(b.colour, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
  }

  degrade() {
    if (!this.quality) { return false; }
    this.quality = 0;
    this.seed();
    this.resize();
    return true;
  }

  clear() {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}

/* #RRGGBB -> rgba(). The input is always a token value, never a literal, so
 * this cannot introduce an unapproved colour. */
function hexA(hex, a) {
  const h = hex.replace("#", "").trim();
  const n = h.length === 3
    ? h.split("").map((c) => parseInt(c + c, 16))
    : [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  return "rgba(" + n[0] + "," + n[1] + "," + n[2] + "," + a + ")";
}

/* ====================================================================
 * 3. Orchestration: one rAF loop, paused when hidden or offscreen
 * ==================================================================== */

const state = {
  ambient: null, hero: null, raf: 0,
  heroVisible: true, running: false,
  samples: [], degraded: false, last: 0,
  heroStatus: "not attempted",
};

/* Checks that a WebGL 2 context can actually be obtained, not merely that the
 * constructor exists: a browser can expose it and still refuse the context on
 * a blocklisted driver or when too many contexts are live. */
function webglOk() {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2", { failIfMajorPerformanceCaveat: false });
    if (!gl) { return false; }
    const lose = gl.getExtension("WEBGL_lose_context");
    if (lose) { lose.loseContext(); }
    return true;
  } catch (e) {
    return false;
  }
}

/* The only place three.js is loaded. 2.1 MB stays unrequested unless both
 * conditions hold, so reduced motion or no WebGL never pays for it. */
async function loadHero(stage) {
  if (state.hero || state.heroStatus === "loading") { return; }
  if (reduceMotion.matches) { state.heroStatus = "skipped: reduced motion"; return; }
  if (!webglOk()) { state.heroStatus = "skipped: no WebGL 2"; return; }
  state.heroStatus = "loading";
  try {
    const mod = await import("./hero3d.js");
    const hero = await mod.mount(stage, { palette, token });
    // Mounted into #prism-bg, which is position:fixed and full-viewport.
    state.hero = hero;
    state.heroStatus = "live";
    root.setAttribute("data-hero", "gl");
    // The SVG stays in the DOM as the fallback and as the thing the 3D is
    // fitted to; CSS hides it only once the GL canvas is really up.
    window.__prismHero = hero;
  } catch (e) {
    state.heroStatus = "failed: " + (e && e.message ? e.message : e);
    root.setAttribute("data-hero", "svg");
    // Deliberately not rethrown. The mark is already on screen in SVG.
    if (window.console) { console.warn("[PrismFlow] 3D hero unavailable:", e); }
  }
}

function loop(t) {
  if (!state.running) { return; }
  if (state.last) { state.samples.push(t - state.last); }
  state.last = t;

  if (state.ambient) { state.ambient.frame(t); }
  // The GL prism is the most expensive thing on the page. It is a fixed
  // background, so "on screen" is always true and the pause that matters is
  // the hidden-tab one in boot().
  if (state.hero && state.heroVisible) { state.hero.frame(t); }

  // Self-degrade: if the median frame over a window misses the budget, drop
  // to fewer blobs and a lower DPR once, then stop measuring.
  if (state.samples.length >= 90 && !state.degraded) {
    const s = state.samples.slice().sort((a, b) => a - b);
    const median = s[Math.floor(s.length / 2)];
    if (median > 20 && state.ambient && state.ambient.degrade()) {
      state.degraded = true;
      document.documentElement.setAttribute("data-motion-degraded", "true");
    }
    state.samples.length = 0;
  }
  state.raf = requestAnimationFrame(loop);
}

function start() {
  if (state.running || reduceMotion.matches) { return; }
  state.running = true; state.last = 0;
  state.raf = requestAnimationFrame(loop);
}

function stop() {
  state.running = false;
  cancelAnimationFrame(state.raf);
}

/* ====================================================================
 * 4. Scroll reveals via anime.js
 * ==================================================================== */

function setupReveals() {
  if (reduceMotion.matches) { return; }

  /* reveal-failsafe.js owns both the attribute that hides and the deadline
   * that clears it. If it did not load, arm() is unreachable and nothing is
   * ever hidden -- we lose the fade, never the content. */
  const failsafe = window.__prismReveal;
  if (!failsafe || !failsafe.arm()) { return; }

  const groups = [
    { sel: ".chapter__lede", d: 0 },
    { sel: ".panel", d: 40 },
    { sel: ".limit", d: 14 },
  ];
  const seen = new WeakSet();
  const io = new IntersectionObserver((entries) => {
    const batch = entries.filter((e) => e.isIntersecting && !seen.has(e.target));
    if (!batch.length) { return; }
    const els = batch.map((e) => e.target);
    const group = groups.find((g) => els[0].matches(g.sel)) || groups[0];
    try {
      animate(els, {
        opacity: [0, 1],
        translateY: [12, 0],
        duration: 520,
        delay: stagger(group.d, { start: 0 }),
        ease: "outQuad",
      });
      /* Marked seen only after animating. Marking first meant a throw inside
       * anime.js retired elements while they were still invisible, and any
       * later sweep skipped them precisely because they were "seen". */
      els.forEach((el) => seen.add(el));
    } catch (err) {
      /* A failed animation costs the fade, never the content. */
      els.forEach((el) => { el.style.opacity = "1"; el.style.transform = "none"; });
      if (window.console) { console.warn("reveal animation failed:", err && err.message); }
    }
    batch.forEach((e) => io.unobserve(e.target));
  }, { rootMargin: "0px 0px -6% 0px", threshold: 0.05 });

  groups.forEach((g) => {
    document.querySelectorAll(g.sel).forEach((el) => io.observe(el));
  });

  /* No safety net here, deliberately. The deadline lives in
   * reveal-failsafe.js, armed before this function runs and firing whatever
   * happens inside it. A second, weaker net would only make it unclear which
   * one is the guarantee. */
}

/* ====================================================================
 * 5. Boot
 * ==================================================================== */

function boot() {
  const stage = document.getElementById("prism-bg");
  const canvas = document.getElementById("ambient");

  if (reduceMotion.matches) {
    // Static everything. The background keeps its static SVG prism, the
    // ambient canvas is never even sized, and no observer is attached.
    if (canvas) { canvas.remove(); }
    root.setAttribute("data-motion", "static");
    return;
  }

  root.setAttribute("data-motion", "live");
  if (canvas) { state.ambient = new Ambient(canvas); }
  setupReveals();

  /* The background is fixed and on screen from first paint, so there is
   * nothing for a viewport observer to wait for. The import is async and
   * gated inside loadHero, so it does not block paint. */
  if (stage) { loadHero(stage); }

  // Pause entirely when the tab is hidden -- rAF is throttled there anyway,
  // and anything still scheduled is pure waste.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { stop(); } else { start(); }
  });

  start();
}

// Exposed for the measurement harness; not used by the page.
window.__prismMotion = {
  start, stop, state,
  isLive: () => state.running,
  frameSamples: () => state.samples.slice(),
  heroStatus: () => state.heroStatus,
  heroInfo: () => (state.hero ? state.hero.info : null),
};

reduceMotion.addEventListener("change", () => window.location.reload());

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
