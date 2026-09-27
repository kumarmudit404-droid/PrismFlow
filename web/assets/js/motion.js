/* PrismFlow Part 25 -- phase (c). Motion, the prism mark, ambient background.
 *
 * PRESENTATION ONLY. Nothing here reads, writes, formats or animates a DATA
 * value. Panels fade and slide; the numbers inside them are already in the DOM
 * at their committed values before any animation starts and are never tweened.
 * A "not measured" cell has no numeric state to animate towards, by
 * construction -- there is no code path that could turn one into a number.
 *
 * THERE IS NO 3D HERO. One was attempted in phase (c) and dropped -- see the
 * hero comment in index.html. The prism mark is SVG, and this file only moves
 * it. Nothing here imports three.js, and nothing loads a WebGL context.
 *
 * COLOUR. Every colour is read from tokens.css at runtime via
 * getComputedStyle. No hex literal appears in this file, so the palette cannot
 * drift from the approved tokens and a change to tokens.css changes the motion
 * too. There is no hue-rotate and no generated hue anywhere.
 */

import { animate, stagger } from "../vendor/anime/anime.esm.min.js";

const root = document.documentElement;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

/* Token reader. If a token is missing we return null rather than a guessed
 * fallback colour, and the caller skips that beam -- inventing a colour here
 * would be exactly the palette drift this indirection exists to prevent. */
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
 * 1. The prism mark -- cursor reactive
 * ==================================================================== */

class PrismMark {
  constructor(svg) {
    this.svg = svg;
    this.beams = Array.from(svg.querySelectorAll("[data-beam]"));
    this.trails = Array.from(svg.querySelectorAll("[data-trail]"));
    this.apex = { x: 400, y: 120 };
    this.target = 0;      // -1 .. 1, pointer position mapped to beam fan
    this.current = 0;
    this.spread = 1;
    this.targetSpread = 1;
    this.idle = true;
    this.history = [];
    this.running = false;
    this.bind();
  }

  bind() {
    const onMove = (clientX, clientY) => {
      const r = this.svg.getBoundingClientRect();
      if (!r.width || !r.height) { return; }
      // -1..1 across the mark, clamped so a pointer far outside does not
      // fling the beams past the frame
      this.target = Math.max(-1, Math.min(1, ((clientY - r.top) / r.height - 0.5) * 2));
      this.targetSpread = 0.55 + Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * 1.1;
      this.idle = false;
      this.lastInput = performance.now();
    };

    window.addEventListener("pointermove", (e) => onMove(e.clientX, e.clientY), { passive: true });
    // Touch: dragging drives the same mapping, so a device with no cursor is
    // not left with a dead mark.
    window.addEventListener("touchmove", (e) => {
      if (e.touches && e.touches[0]) { onMove(e.touches[0].clientX, e.touches[0].clientY); }
    }, { passive: true });
  }

  /* Idle drift: with no pointer (or none for a while) the mark breathes
   * slowly instead of freezing, which is what a touch device sees. */
  idleValue(t) {
    return Math.sin(t / 2600) * 0.45 + Math.sin(t / 4100) * 0.2;
  }

  frame(t) {
    if (this.lastInput && t - this.lastInput > 2500) { this.idle = true; }
    const want = this.idle ? this.idleValue(t) : this.target;
    const wantSpread = this.idle ? 0.9 + Math.sin(t / 3300) * 0.18 : this.targetSpread;

    // critically-damped-ish easing so the beams feel like glass, not elastic
    this.current += (want - this.current) * 0.075;
    this.spread += (wantSpread - this.spread) * 0.06;

    this.history.push(this.current);
    if (this.history.length > 14) { this.history.shift(); }

    const n = this.beams.length;
    this.beams.forEach((beam, i) => {
      const rank = n === 1 ? 0 : (i / (n - 1)) * 2 - 1;   // -1..1 across the fan
      const y = this.apex.y + rank * 82 * this.spread + this.current * 46;
      beam.setAttribute("y2", y.toFixed(2));
    });

    // Spectral trail: older pointer positions, faded. Same beam colours, so
    // the trail can never introduce a hue that is not a token.
    this.trails.forEach((trail, i) => {
      const idx = this.history.length - 1 - (i + 1) * 3;
      if (idx < 0) { return; }
      const past = this.history[idx];
      const rank = (i / Math.max(1, this.trails.length - 1)) * 2 - 1;
      const y = this.apex.y + rank * 82 * this.spread + past * 46;
      trail.setAttribute("y2", y.toFixed(2));
      trail.setAttribute("opacity", (0.16 - i * 0.03).toFixed(3));
    });
  }

  reset() {
    // The static pose: exactly the phase-(a) fan, so reduced-motion users and
    // no-JS users see the identical mark.
    const n = this.beams.length;
    this.beams.forEach((beam, i) => {
      const rank = n === 1 ? 0 : (i / (n - 1)) * 2 - 1;
      beam.setAttribute("y2", (this.apex.y + rank * 82).toFixed(2));
    });
    this.trails.forEach((t) => t.setAttribute("opacity", "0"));
  }
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

      // The unbuilt angle stays dim in the background too. Regulatory must
      // not look alive anywhere on this site, including here.
      //
      // These two numbers are NOT taste. They are solved against the contrast
      // budget: with 'lighter' compositing every blob can overlap every other,
      // so the worst case is all five stacked on one pixel. At 0.11 that put
      // the background under body text at rgb(53,56,67), which dropped
      // --text-muted from 4.97:1 to 2.94:1 and --angle-regulatory from 3.54:1
      // to 2.09:1 -- both below their thresholds. At 0.05 (0.023 for the
      // unbuilt angle), under the 0.84 scrim, the worst case is rgb(21,22,29)
      // and every token still passes. See web/README.md.
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

/* #RRGGBB -> rgba(). Kept tiny and local; the INPUT is always a token value,
 * never a literal, so this cannot introduce an unapproved colour. */
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
  prism: null, ambient: null, raf: 0,
  heroVisible: true, running: false,
  samples: [], degraded: false, last: 0,
};

function loop(t) {
  if (!state.running) { return; }
  if (state.last) { state.samples.push(t - state.last); }
  state.last = t;

  if (state.ambient) { state.ambient.frame(t); }
  if (state.prism && state.heroVisible) { state.prism.frame(t); }

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

  /* Hiding content requires the un-hider. reveal-failsafe.js owns both the
   * attribute that hides and the deadline that clears it; if it did not load,
   * `arm()` is unreachable and nothing is ever hidden. Losing the fade is an
   * acceptable degradation. Losing the page is not, and that is exactly what
   * happened when this function armed the hide itself. */
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
      /* Marked seen only now. Marking before the call meant a throw inside
       * anime.js retired these elements while they were still invisible, and
       * any later sweep would skip them precisely because they were "seen". */
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

  /* The safety net used to live here. It does not any more, and deliberately:
   * scheduled from inside this function, it was never reached when boot() threw
   * before calling it, and it skipped anything already marked "seen". Both
   * failure modes left real content permanently invisible. The deadline now
   * lives in reveal-failsafe.js, armed before this function runs and running
   * whatever happens in here. Nothing replaces it at this level -- a second,
   * weaker net would only make it unclear which one is the guarantee. */
}

/* ====================================================================
 * 5. Boot
 * ==================================================================== */

function boot() {
  const svg = document.getElementById("prism-mark");
  const canvas = document.getElementById("ambient");

  if (svg) {
    state.prism = new PrismMark(svg);
    state.prism.reset();
  }

  if (reduceMotion.matches) {
    // Static everything. The prism keeps its phase-(a) pose, the canvas is
    // never even sized, and no observer is attached.
    if (canvas) { canvas.remove(); }
    root.setAttribute("data-motion", "static");
    return;
  }

  root.setAttribute("data-motion", "live");
  if (canvas) { state.ambient = new Ambient(canvas); }
  setupReveals();

  // Pause when the hero leaves the viewport: the prism is the only thing that
  // needs per-frame work tied to an element.
  if (svg && "IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      state.heroVisible = entries[0].isIntersecting;
    }, { threshold: 0.01 }).observe(svg);
  }

  // Pause entirely when the tab is hidden -- rAF is throttled there anyway,
  // and anything still scheduled is pure waste.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { stop(); } else { start(); }
  });

  start();
}

// Exposed for the phase-(d)-style measurement harness, not used by the page.
window.__prismMotion = {
  start, stop, state,
  isLive: () => state.running,
  frameSamples: () => state.samples.slice(),
};

reduceMotion.addEventListener("change", () => window.location.reload());

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
