/* PrismFlow Part 25 -- phase (c). Motion, the prism mark, ambient background.
 *
 * PRESENTATION ONLY. Nothing here reads, writes, formats or animates a DATA
 * value. Panels fade and slide; the numbers inside them are already in the DOM
 * at their committed values before any animation starts and are never tweened.
 * A "not measured" cell has no numeric state to animate towards, by
 * construction -- there is no code path that could turn one into a number.
 *
 * THE 3D PRISM IS OPTIONAL. Phase (c) attempted one and dropped it because it
 * rendered as a grey slab; phase (f) fixed the cause -- see the top of
 * hero3d.js -- and phase (g) promoted it from the hero slot to the site-wide
 * fixed background in #prism-bg. Nothing about the page depends on it.
 * three.js is imported dynamically, only once WebGL 2 is actually available AND
 * motion is not reduced. The background is at the top of the page and visible
 * immediately, so there is no viewport trigger left to wait for -- the import
 * is still async and off the critical path. If it fails, the context is lost,
 * or the module throws for any reason at all, the static SVG mark in #prism-bg
 * stays exactly where it is and the page is unchanged.
 *
 * COLOUR. Every colour is read from tokens.css at runtime via
 * getComputedStyle. No hex literal appears in this file, so the palette cannot
 * drift from the approved tokens and a change to tokens.css changes the motion
 * too. There is no hue-rotate and no generated hue anywhere.
 */

import { animate, stagger } from "../vendor/anime/anime.esm.min.js";

const root = document.documentElement;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

/* Token reader. A missing token returns null rather than a guessed fallback
 * colour, and the caller skips that beam -- inventing a colour here would be
 * exactly the palette drift this indirection exists to prevent. */
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
 * 1. (was the cursor-reactive prism mark)
 * ====================================================================
 * Phase (g) removed it. It animated [data-beam] lines inside the hero SVG;
 * the mark is now the fixed site-wide background and is deliberately STATIC
 * there, so the class had no host left and is deleted rather than kept as a
 * no-op. The motion the prism does have lives in hero3d.js, behind the pauses
 * in section 3.
 */

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
  ambient: null, hero: null, raf: 0,
  heroVisible: true, running: false,
  samples: [], degraded: false, last: 0,
  heroStatus: "not attempted",
};

/* WebGL 2, actually obtained -- not "is WebGL2RenderingContext defined". A
 * browser can expose the constructor and still refuse the context on a
 * blocklisted driver, in a low-power state, or once too many contexts are
 * live. The probe canvas is discarded immediately. */
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

/* The only place three.js is ever loaded. 2.1 MB stays unrequested unless both
 * conditions hold, so a reader with reduced motion on, or whose machine has no
 * WebGL, never pays for it. */
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
    // fitted to; it is hidden by CSS only once the GL canvas is really up.
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
  // The GL prism is the most expensive thing on the page. It is now a fixed
  // background, so "on screen" is always true and the pauses that matter are
  // the hidden-tab one in boot() and, from step 5, idle.
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

  /* The background is fixed and on screen from the first paint, so the phase
   * (f) pair of IntersectionObservers -- one to pause the hero offscreen, one
   * to pre-load it just before it scrolled in -- have nothing left to observe.
   * The import is started here instead. It is async and gated on WebGL 2 and
   * reduced motion inside loadHero, so nothing about this blocks paint. */
  if (stage) { loadHero(stage); }

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
  heroStatus: () => state.heroStatus,
  heroInfo: () => (state.hero ? state.hero.info : null),
};

reduceMotion.addEventListener("change", () => window.location.reload());

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
