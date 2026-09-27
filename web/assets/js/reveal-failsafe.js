/* PrismFlow Part 25 -- phase (c). The reveal failsafe.
 *
 * WHY THIS IS ITS OWN FILE
 * The scroll reveals hide real content -- chapter ledes, every panel, the whole
 * limitations hub -- so it can fade in. Hiding content is only defensible if
 * something guarantees it comes back. That guarantee cannot live in motion.js,
 * because the failure it exists to survive IS motion.js dying, and a net inside
 * the thing it protects against is not a net.
 *
 * That is measured, not assumed. With the net inside motion.js, a throw placed
 * immediately after it set data-motion="live" left all 67 reveal targets at
 * computed opacity 0 permanently -- 9s after load, well past the 5s net, which
 * was never reached because it was scheduled further down the same function.
 * A second hole: the observer marked elements "seen" BEFORE animating them, so
 * a throw inside anime.js left 4 more hidden even when the net did run.
 *
 * THE CONTRACT, IN THREE LINES
 *   1. Content is visible by default. CSS hides a reveal target only while
 *      <html data-reveal="armed">.
 *   2. motion.js may arm that attribute ONLY through this file. No un-hider
 *      loaded, no hiding -- so a missing or broken failsafe costs the fade,
 *      never the content.
 *   3. This deadline runs from the moment THIS file executes and depends on
 *      nothing else: not motion.js, not anime.js, not WebGL, not an observer.
 *      When it fires, the page is readable whatever happened elsewhere.
 *
 * It is a classic script, not a module, and it is loaded in <head> ahead of
 * everything else, so it has already run before anything can ask to hide.
 */
(function () {
  "use strict";

  var SELECTORS = [".chapter__lede", ".panel", ".limit"];

  /* 5000ms preserves the budget phase (c) already chose. It is a deadline for
   * the whole page, not per element: past it, everything is simply shown. */
  var DEADLINE_MS = 5000;

  var root = document.documentElement;
  var released = false;

  function releaseAll() {
    if (released) { return; }
    released = true;

    root.removeAttribute("data-reveal");

    /* Dropping the attribute is not sufficient on its own. An animation that
     * died mid-flight can leave inline opacity behind, and an inline style
     * outranks the stylesheet -- so the element would stay invisible with the
     * hide rule already gone. Clear those too. */
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
    /* The ONLY way content can be hidden. motion.js calls this; nothing else
     * should. Once the deadline has passed, arming is refused outright rather
     * than hiding content the deadline has already promised to show. */
    arm: function () {
      if (!released) { root.setAttribute("data-reveal", "armed"); }
      return !released;
    },
    /* Called when the reveals finish normally, so the attribute does not sit
     * on the page for longer than it is doing anything. */
    release: releaseAll,
    deadline: DEADLINE_MS,
    /* Read by the phase-(d)-style harness; not used by the page. */
    isReleased: function () { return released; }
  };

  window.setTimeout(releaseAll, DEADLINE_MS);
})();
