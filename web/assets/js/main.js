/* PrismFlow Part 25 -- phase (a) behaviour only.
 *
 * Deliberately tiny: nav state and a reduced-motion-aware reveal. anime.js and
 * the ray-traced hero arrive in phase (c). Nothing here fetches anything, and
 * no credential of any kind is referenced -- this site reads committed files
 * and nothing else.
 */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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

  /* ---- staggered reveal ----------------------------------------------
   * Content is visible by default in the CSS; this only ADDS a fade when
   * motion is welcome. If JS fails or motion is reduced, the page is simply
   * already readable -- the reveal is never load-bearing. */
  if (!reduceMotion && "IntersectionObserver" in window) {
    var targets = document.querySelectorAll(".panel, .chapter__lede, .limit");
    Array.prototype.forEach.call(targets, function (el) {
      el.style.opacity = "0";
      el.style.transform = "translateY(10px)";
      el.style.transition = "opacity .5s ease, transform .5s ease";
    });
    var rev = new IntersectionObserver(function (entries, o) {
      entries.forEach(function (e, i) {
        if (!e.isIntersecting) { return; }
        var el = e.target;
        setTimeout(function () {
          el.style.opacity = "1";
          el.style.transform = "none";
        }, Math.min(i * 45, 220));
        o.unobserve(el);
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    Array.prototype.forEach.call(targets, function (el) { rev.observe(el); });

    /* Safety net: if anything above misbehaves, nothing stays invisible. */
    window.setTimeout(function () {
      Array.prototype.forEach.call(targets, function (el) {
        if (el.style.opacity === "0") { el.style.opacity = "1"; el.style.transform = "none"; }
      });
    }, 4000);
  }
})();
