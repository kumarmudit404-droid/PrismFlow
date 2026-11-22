/* Motion: the 3D prism, the scroll reveals and the progress line.
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
 * 2. The ambient canvas: REMOVED
 * ====================================================================
 * There used to be a full-viewport 2D canvas here painting five drifting
 * radial gradients in the angle colours, with its own DPR cap, a frame-budget
 * self-degrade and a blob-count fallback.
 *
 * The backdrop is an OPAQUE full-viewport photograph now, fixed at z-index -1,
 * and the canvas sat at -3. It was painting five full-screen gradients every
 * frame, forever, under something that covered every one of its pixels. The
 * two body pseudo-layers it shared that space with -- the spectrum wash and
 * the scrim -- are gone from app.css for the same reason.
 *
 * Nothing replaced it: the drifting colour it provided is in the photograph.
 * ==================================================================== */

/* ====================================================================
 * 3. Orchestration: one rAF loop, paused when hidden or offscreen
 * ==================================================================== */

const state = {
  hero: null, raf: 0,
  heroVisible: true, running: false,
  samples: [], last: 0,
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

  // The GL prism is the only thing this loop drives. It is a fixed background
  // and therefore always on screen, so the pause that matters is the
  // hidden-tab one in boot() -- see the note there.
  if (state.hero && state.heroVisible) { state.hero.frame(t); }

  // Kept only to feed the measurement harness. The self-degrade that used to
  // read it belonged to the ambient canvas and went with it; the prism has
  // its own DPR cap in hero3d.js's resize().
  if (state.samples.length >= 180) { state.samples.length = 0; }
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
 * 4. Scroll reveals, as a CSS class
 * ==================================================================== */

function setupReveals() {
  if (reduceMotion.matches) { return; }

  /* reveal-failsafe.js owns both the attribute that hides and the deadline
   * that clears it. If it did not load, arm() is unreachable and nothing is
   * ever hidden -- we lose the fade, never the content. */
  const failsafe = window.__prismReveal;
  if (!failsafe || !failsafe.arm()) { return; }

  /* The work is one class. The 560ms curve, the 20px offset and the 80ms
   * stagger are all in app.css, so a throw in here cannot leave an element
   * part-way through a tween the way a JS-driven opacity could. */
  const els = document.querySelectorAll(
    ".reveal, .chapter__lede, .panel, .limit");
  if (!els.length) { return; }

  const io = new IntersectionObserver((entries) => {
    let i = 0;
    entries.forEach((e) => {
      if (!e.isIntersecting) { return; }
      /* --i staggers within THIS batch, not within the document: elements
       * that scroll into view together are the ones that should cascade. */
      e.target.style.setProperty("--i", i++);
      e.target.classList.add("is-visible");
      io.unobserve(e.target);          // observe once, then stop
    });
  }, { threshold: 0.15, rootMargin: "0px 0px -10% 0px" });

  els.forEach((el) => io.observe(el));

  /* No safety net here, deliberately. The deadline lives in
   * reveal-failsafe.js, armed before this function runs and firing whatever
   * happens inside it. A second, weaker net would only make it unclear which
   * one is the guarantee. */
}

/* ====================================================================
 * 4b. The scroll-progress line
 * ==================================================================== */

function setupProgress() {
  const bar = document.querySelector(".scroll-progress");
  if (!bar) { return; }
  /* Static under reduced motion: the line is decoration and duplicates the
   * scrollbar, so it is left at zero rather than animated. */
  if (reduceMotion.matches) { return; }

  let queued = false;
  function update() {
    queued = false;
    const doc = document.documentElement;
    const span = doc.scrollHeight - doc.clientHeight;
    bar.style.setProperty("--p", span > 0 ? Math.min(1, doc.scrollTop / span) : 0);
  }
  // One write per frame. A scroll handler that writes a style every event is
  // a layout thrash on a long page.
  window.addEventListener("scroll", () => {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, { passive: true });
  window.addEventListener("resize", update, { passive: true });
  update();
}

/* ====================================================================
 * 5. Boot
 * ==================================================================== */

function boot() {
  const stage = document.getElementById("prism-bg");

  if (reduceMotion.matches) {
    // Static everything. The backdrop keeps its static SVG prism and no
    // observer is attached.
    root.setAttribute("data-motion", "static");
    return;
  }

  root.setAttribute("data-motion", "live");
  setupReveals();
  setupProgress();

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
