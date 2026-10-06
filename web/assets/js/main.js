/* Nav state, and keyboard access to the scrollable tables.
 *
 * Deliberately tiny. All reveal and motion behaviour lives in motion.js under
 * the reveal-failsafe contract; nothing here fetches anything.
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
   * An overflow-x:auto wrapper scrolls with a mouse but is unreachable by
   * keyboard unless it is focusable (WCAG 2.1.1). We add tabindex only to
   * wrappers that actually overflow, so a table that fits does not become a
   * dead tab stop, and re-check on resize since overflow depends on width. */
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
