/* PrismFlow Part 25 -- phase (a) behaviour only.
 *
 * Deliberately tiny: nav state, and keyboard access to the scrollable tables.
 * No reveal and no motion of any kind live here -- phase (c) moved all of that
 * to motion.js, under the reveal-failsafe contract. Nothing here fetches
 * anything, and no credential of any kind is referenced -- this site reads
 * committed files and nothing else.
 */
(function () {
  "use strict";

  /* ---- nav: mark the chapter currently in view ------------------------ */
  var links = Array.prototype.slice.call(document.querySelectorAll(".site-nav__link"));
  var sections = links
    .map(function (a) { return document.querySelector(a.getAttribute("href")); })
    .filter(Boolean);

  function setCurrent(id) {
    links.forEach(function (a) {
      var on = a.getAttribute("href") === "#" + id;
      if (on) { a.setAttribute("aria-current", "true"); }
      else { a.removeAttribute("aria-current"); }
    });
  }

  if ("IntersectionObserver" in window && sections.length) {
    var seen = new Map();
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { seen.set(e.target.id, e.intersectionRatio); });
      var best = null, bestRatio = 0;
      seen.forEach(function (ratio, id) {
        if (ratio > bestRatio) { bestRatio = ratio; best = id; }
      });
      if (best && bestRatio > 0) { setCurrent(best); }
    }, { threshold: [0, 0.15, 0.4, 0.75], rootMargin: "-70px 0px -40% 0px" });
    sections.forEach(function (s) { obs.observe(s); });
  }

  /* ---- keyboard access to horizontally scrollable tables ---------------
   * A wrapper with overflow-x:auto scrolls with a mouse or a trackpad but is
   * unreachable by keyboard unless it is focusable. The 48-row coverage table
   * overflows on a narrow window, so without this a keyboard-only user cannot
   * see its right-hand columns at all (WCAG 2.1.1). tabindex is added ONLY to
   * wrappers that actually overflow, so a table that fits does not become a
   * dead tab stop -- and it is re-evaluated on resize, because whether a table
   * overflows is a function of the viewport. */
  function updateScrollRegions() {
    var wraps = document.querySelectorAll(".table-wrap");
    Array.prototype.forEach.call(wraps, function (w) {
      var overflows = w.scrollWidth > w.clientWidth + 1;
      if (overflows) {
        if (!w.hasAttribute("tabindex")) {
          w.setAttribute("tabindex", "0");
          w.setAttribute("role", "region");
          var table = w.querySelector("table");
          var cap = table && table.querySelector("caption");
          w.setAttribute("aria-label",
            (cap ? cap.textContent.trim().slice(0, 80) : "Data table") +
            " (scrollable — use arrow keys)");
        }
      } else if (w.hasAttribute("tabindex")) {
        w.removeAttribute("tabindex");
        w.removeAttribute("role");
        w.removeAttribute("aria-label");
      }
    });
  }

  // Charts render asynchronously, so run after they land as well as on resize.
  window.addEventListener("load", updateScrollRegions);
  window.addEventListener("resize", updateScrollRegions);
  setTimeout(updateScrollRegions, 600);
  setTimeout(updateScrollRegions, 1800);
})();
