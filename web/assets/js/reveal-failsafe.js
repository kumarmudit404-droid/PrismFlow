/* The reveal failsafe: guarantees hidden content always comes back.
 *
 * The scroll reveals hide real content so it can fade in, so something has to
 * guarantee it reappears. That guarantee lives here rather than in motion.js,
 * because the failure it exists to survive is motion.js itself dying.
 *
 * The contract:
 *   1. Content is visible by default. CSS hides a reveal target only while
 *      <html data-reveal="armed">.
 *   2. motion.js may set that attribute only through this file, so a missing
 *      or broken failsafe costs the fade, never the content.
 *   3. The deadline below runs from the moment this file executes and depends
 *      on nothing else -- not motion.js, WebGL or an observer.
 *
 * Classic script, not a module, loaded in <head> ahead of everything else so
 * it has already run before anything can ask to hide.
 */
(function () {
  "use strict";

  var SELECTORS = [".reveal", ".chapter__lede", ".panel", ".limit"];

  /* A deadline for the whole page, not per element. */
  var DEADLINE_MS = 5000;

  var root = document.documentElement;
  var released = false;

  function releaseAll() {
    if (released) { return; }
    released = true;

    root.removeAttribute("data-reveal");

    /* Dropping the attribute is not enough on its own: an animation that died
     * mid-flight can leave inline opacity behind, and that outranks the
     * stylesheet, so the element would stay invisible with the hide rule
     * already gone. */
    for (var i = 0; i < SELECTORS.length; i++) {
      var els = document.querySelectorAll(SELECTORS[i]);
      for (var j = 0; j < els.length; j++) {
        var el = els[j];
        if (el.style.opacity !== "" && parseFloat(el.style.opacity) < 1) {
          el.style.opacity = "1";
          el.style.transform = "none";
        }
      }
    }
  }

  window.__prismReveal = {
    /* The only way content can be hidden. Past the deadline, arming is refused
     * rather than hiding content we have already promised to show. */
    arm: function () {
      if (!released) { root.setAttribute("data-reveal", "armed"); }
      return !released;
    },
    /* Called when the reveals finish normally, so the attribute does not
     * outlive its purpose. */
    release: releaseAll,
    deadline: DEADLINE_MS,
    isReleased: function () { return released; }
  };

  window.setTimeout(releaseAll, DEADLINE_MS);
})();
