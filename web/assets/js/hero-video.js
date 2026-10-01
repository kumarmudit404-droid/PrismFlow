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
    playError: null,
    observer: null,
    onScreen: null
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

  /* ==================================================================
   * Pause off-screen -- phase (g) step 8.4
   * ==================================================================
   * The hero is at the top of the page and a reader scrolls past it within
   * seconds, after which the element is decoding 14.29 fps of 1072x368 that
   * nobody can see. motion.js retired its own pair of observers when the
   * prism became a fixed background (see its boot comment); this is the one
   * that comes back with the video, and it observes the video alone.
   *
   * Only under `live`. Under reduced motion there is nothing to pause, and
   * resuming on scroll would be precisely the autoplay the gate above exists
   * to prevent -- so the observer is never attached in that branch rather
   * than attached and guarded, which would leave one `play()` call reachable.
   *
   * threshold 0 with a 64px bottom margin: the state flips as the last of the
   * element leaves the viewport, slightly late rather than slightly early, so
   * a reader who scrolls a little and comes back finds it still running.
   */
  function observe() {
    if (info.reduced) { return; }
    if (typeof IntersectionObserver !== "function") {
      info.observer = "unsupported: plays continuously";
      return;
    }

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        info.onScreen = entry.isIntersecting;
        if (entry.isIntersecting) {
          /* Never resume what the engine refused to start, and never resume
           * what the reader's own setting stopped. Resuming a blocked video
           * on scroll would be a second, quieter autoplay attempt. */
          if (info.played === false) { return; }
          var p = video.play();
          if (p && typeof p.catch === "function") {
            p.catch(function () { /* still refused; poster stands */ });
          }
        } else {
          video.pause();
        }
      });
    }, { threshold: 0, rootMargin: "0px 0px 64px 0px" });

    io.observe(video);
    info.observer = "attached";
  }

  observe();

  /* The tab being hidden is the same waste as the hero being off-screen, and
   * is not something an IntersectionObserver reports -- an element in a
   * backgrounded tab is still "intersecting". motion.js takes the same pair
   * for its canvas. */
  if (!info.reduced) {
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        video.pause();
      } else if (info.onScreen !== false && info.played !== false) {
        var p = video.play();
        if (p && typeof p.catch === "function") { p.catch(function () {}); }
      }
    });
  }

  window.__prismHeroVideo = {
    info: function () { return info; },
    el: function () { return video; }
  };
}());
