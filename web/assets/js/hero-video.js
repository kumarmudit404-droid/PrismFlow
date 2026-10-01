/* Hero video -- reduced-motion gate.
 * ====================================================================
 * Phase (g) step 8. Governs the <video> in the hero slot and nothing else.
 *
 * WHY THIS IS A CLASSIC SCRIPT SITTING DIRECTLY AFTER THE ELEMENT
 * ---------------------------------------------------------------
 * `autoplay` is in the markup, which means the engine may begin playback the
 * moment it has enough data -- it does not wait for DOMContentLoaded. A
 * `defer`red script, or anything at the foot of the body, is therefore racing
 * the network: it would win almost every time and lose exactly when the file
 * is warm in cache, which is the common case on a second visit and the
 * guaranteed case on the local server this is developed against.
 *
 * Running inline-position (classic, no defer, immediately after the element)
 * means this executes while the parser is still inside <header>, before the
 * element can have buffered a frame. That is the only placement that makes
 * "reduced motion does not autoplay" a property of the page rather than a
 * race it usually wins.
 *
 * The alternative -- drop `autoplay` from the markup and call play() here --
 * was rejected: it moves the decision out of the markup, so the element no
 * longer declares its own behaviour, and a reader with JS disabled gets a
 * dead frame instead of a playing hero.
 *
 * WHY motion.js DOES NOT OWN THIS
 * -------------------------------
 * motion.js returns early under reduced motion (`boot`, the first branch), by
 * design -- it is the module that must not run. The video needs the opposite:
 * code that runs PRECISELY when motion is reduced, to take the autoplay away.
 * Putting that inside the module that early-returns would have meant moving
 * its guard, and motion.js already owns the reload-on-change for the whole
 * page, which covers this element too.
 */
(function () {
  "use strict";

  var video = document.getElementById("hero-video");
  if (!video) { return; }

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* Reported to the phase-(d)-style harness, not read by the page. `played`
   * stays null until the promise settles, so "not yet known" is a distinct
   * state from "blocked" -- the measurement in step 6 needs that distinction,
   * because a pending promise and a rejected one look identical in a
   * screenshot. */
  var info = {
    reduced: reduceMotion.matches,
    mode: null,
    played: null,
    playError: null
  };

  /* Take the autoplay away and sit on the poster.
   *
   * Called twice under reduced motion, deliberately. A play() promise issued
   * by the engine's own autoplay logic before this ran can still resolve
   * AFTER our pause() and restart playback, so the pause is repeated once
   * metadata has arrived -- the earliest point at which the element is
   * guaranteed to honour it. */
  function freeze() {
    video.autoplay = false;
    video.removeAttribute("autoplay");
    video.loop = false;
    try { video.pause(); } catch (err) { /* not fatal: the poster still shows */ }
    try { video.currentTime = 0; } catch (err) { /* no metadata yet; fine */ }
  }

  if (info.reduced) {
    info.mode = "static: reduced motion";
    freeze();
    video.addEventListener("loadedmetadata", freeze, { once: true });
    document.documentElement.setAttribute("data-hero-video", "static");
  } else {
    info.mode = "live";
    document.documentElement.setAttribute("data-hero-video", "live");

    /* Whether autoplay was actually permitted is only knowable from the
     * promise. Engines reject it silently; without this the slot would show a
     * still poster and nothing would say why. */
    video.addEventListener("loadedmetadata", function () {
      var p = video.play();
      if (!p || typeof p.then !== "function") { info.played = true; return; }
      p.then(function () {
        info.played = true;
      }, function (err) {
        info.played = false;
        info.playError = (err && err.name) ? err.name : String(err);
        /* Autoplay refused. The poster is already showing, so the hero
         * degrades to the still frame rather than to an empty box. */
        document.documentElement.setAttribute("data-hero-video", "blocked");
      });
    }, { once: true });
  }

  window.__prismHeroVideo = {
    info: function () { return info; },
    el: function () { return video; }
  };
}());
