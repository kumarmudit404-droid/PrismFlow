r"""Measure text contrast over the live prism background. Real pixels, real GPU.

    .venv\Scripts\python.exe web/check_prism_contrast.py
    .venv\Scripts\python.exe web/check_prism_contrast.py --opacity 0.30

WHY THIS EXISTS
---------------
``--prism-bg-opacity`` trades "the prism is visible" against "the text on top of
it is legible", and that trade cannot be reasoned about from the token values:
the prism is an additive, animated, tone-mapped render, and what lands under a
paragraph is a composite of it, the body::before wash, the scrim and the page
base. check_palette.py measures the PALETTE; this measures the PAGE.

HOW IT MEASURES
---------------
1. The real site, in the real browser, on the real adapter (``Browser(gpu=True)``
   from verify_page.py -- not SwiftShader).
2. Text is made ``color: transparent``, NOT hidden. Glyphs disappear and every
   element keeps its own background, so what is sampled is exactly the pixels
   that would sit behind and between the glyphs.
3. The rAF loop is stopped and ``hero.frame(t)`` is driven directly across a
   full rotation cycle, so the sample covers the prism's BRIGHTEST frames
   rather than whichever frame a screenshot happened to catch. The three
   rotation terms have periods of 28.6s, 37.0s and 57.1s, so t is swept to 60s.
4. For each sampled region the WORST (brightest) background pixel in the
   element's own box across all frames is kept, and the contrast ratio is
   computed against that element's real computed text colour.

The worst pixel, not the mean: a mean would let a bright beam crossing one line
of a paragraph average away against the dark margin beside it.

EXIT CODE is 1 if any sampled region falls below 4.5:1, so this can gate.
"""
from __future__ import annotations

import argparse
import base64
import io as _io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verify_page import BASE, Browser  # noqa: E402

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: it is the only way to read back the "
                     "composited pixels. Nothing is estimated without it.")

# Sampled regions: a selector, and why it is on the list.
REGIONS = [
    ("header.hero .hero__thesis", "hero prose, widest measure, top of page"),
    ("header.hero p.prose", "hero body prose"),
    ("#ch1 .chapter__lede", "chapter opener, large text"),
    ("#ch1 .prose p", "mid-page prose, the most common case"),
    ("#ch3 .prose p", "mid-page prose, further down"),
    ("#ch4 p", "prose immediately before the results views"),
    ("#limits p", "limitations hub prose"),
    (".site-nav__brand", "sticky nav, which the prism scrolls under"),
]

# t sweep, ms. The rotation periods are 28.6s / 37.0s / 57.1s.
FRAMES = [0, 6000, 12000, 18000, 24000, 30000, 36000, 42000, 48000, 54000, 60000]

VIEWPORTS = [(1440, 900, "desktop"), (390, 844, "narrow / phone width")]


def srgb_lum(rgb) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(rgb[0]) + 0.7152 * ch(rgb[1]) + 0.0722 * ch(rgb[2])


def ratio(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def parse_rgb(css: str):
    body = css.replace("rgba(", "").replace("rgb(", "").replace(")", "")
    nums = [float(n) for n in body.split(",")[:3]]
    return tuple(int(round(n)) for n in nums)


def _shot_crop(b, r):
    """One viewport screenshot, cropped to a region's box. None if off-screen."""
    data = b.ws.call("Page.captureScreenshot", {"format": "png"})["data"]
    im = Image.open(_io.BytesIO(base64.b64decode(data))).convert("RGB")
    x0, y0 = min(r["x"], im.width - 1), min(r["y"], im.height - 1)
    x1 = min(r["x"] + r["w"], im.width)
    y1 = min(r["y"] + r["h"], im.height)
    if x1 <= x0 or y1 <= y0:
        return None
    return im.crop((x0, y0, x1, y1))


PROBE_CSS = (
    "*,*::before,*::after{color:transparent !important;"
    "text-shadow:none !important;-webkit-text-fill-color:transparent !important;}"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--opacity", default=None,
                    help="override --prism-bg-opacity for this measurement "
                         "only; the file on disk is not touched")
    args = ap.parse_args()

    worst_overall = (99.0, "nothing measured")
    failures = []
    measured = 0

    b = Browser(gpu=True)
    try:
        b.goto(BASE + "/", settle=2.0)
        time.sleep(9.0)
        status = b.js("window.__prismMotion ? window.__prismMotion.heroStatus()"
                      " : 'no motion.js'")
        info = b.js("JSON.stringify(window.__prismMotion && "
                    "window.__prismMotion.heroInfo())")
        print("hero status: %s" % status)
        print("adapter: %s" % info)
        if status != "live":
            print("\nThe GL prism is not live, so there is nothing to measure "
                  "against. Not reporting a ratio for a background that did "
                  "not render.")
            return 1

        if args.opacity:
            b.js("document.documentElement.style.setProperty("
                 "'--prism-bg-opacity', %s)" % json.dumps(args.opacity))
        applied = b.js("getComputedStyle(document.documentElement)"
                       ".getPropertyValue('--prism-bg-opacity').trim()")
        print("--prism-bg-opacity under measurement: %s" % applied)
        print("frames swept (ms): %s" % FRAMES)

        # Stop the loop so frame(t) is the only thing driving the prism.
        b.js("window.__prismMotion && window.__prismMotion.stop()")

        # EVERY TEXT COLOUR IS READ FIRST, BEFORE the glyphs are made
        # transparent. Reading it afterwards returns rgba(0,0,0,0) for
        # everything and every ratio computed from it is against black --
        # which is what the first run of this script did.
        colours = {}
        for selector, _why in REGIONS:
            got = b.js("(() => { const e = document.querySelector("
                       + json.dumps(selector) + ");"
                       " return e ? getComputedStyle(e).color : null; })()")
            if got:
                colours[selector] = got
        print("text colours read before the probe: %s"
              % json.dumps(colours, indent=2))

        b.js("(() => { const s = document.createElement('style');"
             " s.id = '__contrast_probe'; s.textContent = " + json.dumps(PROBE_CSS) + ";"
             " document.head.appendChild(s); return 'ok'; })()")

        for vw, vh, vlabel in VIEWPORTS:
            b.ws.call("Emulation.setDeviceMetricsOverride", {
                "width": vw, "height": vh, "deviceScaleFactor": 1,
                "mobile": False})
            time.sleep(1.0)
            print("\n" + "=" * 72)
            print("viewport %dx%d -- %s" % (vw, vh, vlabel))
            print("%-34s %-9s %-10s %-6s %s" % ("region", "worst", "text", "px", "verdict"))

            for selector, why in REGIONS:
                box = b.js(
                    "(() => { const e = document.querySelector("
                    + json.dumps(selector) + "); if (!e) return null;"
                    " e.scrollIntoView({block: 'center'});"
                    " const r = e.getBoundingClientRect();"
                    " return JSON.stringify({x: Math.max(0, Math.floor(r.left)),"
                    " y: Math.max(0, Math.floor(r.top)), w: Math.ceil(r.width),"
                    " h: Math.ceil(r.height)}); })()")
                if not box or selector not in colours:
                    print("%-34s %s" % (selector[:34],
                                        "selector not present -- not measured"))
                    continue
                r = json.loads(box)
                if r["w"] < 2 or r["h"] < 2:
                    print("%-34s %s" % (selector[:34],
                                        "zero-size box -- not measured"))
                    continue
                text_rgb = parse_rgb(colours[selector])
                text_lum = srgb_lum(text_rgb)

                # ONLY THE PIXELS THE PRISM ACTUALLY CHANGES.
                #
                # The first version of this took the brightest pixel in the
                # element's box, which measured the wrong thing: the nav's own
                # logo image is the brightest pixel inside .site-nav__brand and
                # has nothing to do with the prism, so the nav "failed" at
                # 1.12:1 whatever the prism did. The question here is whether
                # the PRISM breaks a ratio, and the page's own composition is
                # already gated by check_palette.py.
                #
                # So the region is captured twice: once with #prism-bg hidden,
                # once with it live. Hiding a position:fixed layer shifts no
                # layout, so the two crops are pixel-aligned, and only the
                # pixels that differ are the prism's. Among those, the
                # brightest wins.
                shot = _shot_crop(b, r)
                if shot is None:
                    print("%-34s %s" % (selector[:34],
                                        "off-screen -- not measured"))
                    continue

                # Let the reveals this scroll just triggered finish. Without
                # this the control below sees anime.js mid-fade, marks the
                # whole block unstable and excludes the entire region -- which
                # is why several regions reported "0 pixels changed".
                time.sleep(1.8)
                b.js("document.getElementById('prism-bg').style.display = 'none'")
                time.sleep(0.35)
                without = _shot_crop(b, r)
                time.sleep(0.35)
                # A CONTROL BASELINE. Two captures with the prism hidden, so
                # any pixel that moves between them is the PAGE moving, not the
                # prism: a reveal transition still settling, the ambient canvas,
                # a late web font. Without this control the diff credited that
                # noise to the prism and reported the identical worst ratio at
                # 0.40 and at 0.20 -- the tell that the number was not
                # responding to the thing being measured.
                control = _shot_crop(b, r)
                b.js("document.getElementById('prism-bg').style.display = ''")
                time.sleep(0.25)
                if without is None or control is None:
                    print("%-34s %s" % (selector[:34],
                                        "off-screen -- not measured"))
                    continue
                base_px = list(without.getdata())
                ctrl_px = list(control.getdata())
                unstable = set()
                if len(ctrl_px) == len(base_px):
                    for i, px in enumerate(ctrl_px):
                        bp = base_px[i]
                        if (abs(px[0] - bp[0]) + abs(px[1] - bp[1])
                                + abs(px[2] - bp[2])) >= 6:
                            unstable.add(i)

                brightest = -1.0
                touched = 0
                for t in FRAMES:
                    b.js("window.__prismHero && window.__prismHero.frame(%d)" % t)
                    crop = _shot_crop(b, r)
                    if crop is None:
                        continue
                    live_px = list(crop.getdata())
                    if len(live_px) != len(base_px):
                        continue
                    for i, px in enumerate(live_px):
                        if i in unstable:
                            continue
                        bp = base_px[i]
                        # A pixel the prism moved at all. 2/255 per channel is
                        # above PNG/compositor rounding and below anything
                        # visible.
                        if (abs(px[0] - bp[0]) + abs(px[1] - bp[1])
                                + abs(px[2] - bp[2])) < 6:
                            continue
                        touched += 1
                        lum = srgb_lum(px)
                        if lum > brightest:
                            brightest = lum

                if brightest < 0:
                    print("%-34s %s" % (selector[:34],
                                        "prism does not reach this region "
                                        "(0 pixels changed) -- nothing to measure"))
                    continue

                cr = ratio(text_lum, brightest)
                ok = cr >= 4.5
                measured += 1
                print("%-34s %-9.2f %-10s %-6s %s"
                      % (selector[:34], cr, "#%02X%02X%02X" % text_rgb,
                         "%d px" % touched,
                         "pass" if ok else "FAIL (< 4.5:1)"))
                if cr < worst_overall[0]:
                    worst_overall = (cr, "%s @ %s" % (selector, vlabel))
                if not ok:
                    failures.append((vlabel, selector, cr))
    finally:
        b.kill()

    print("\n" + "=" * 72)
    print("regions measured: %d" % measured)
    print("worst ratio anywhere: %.2f:1  (%s)" % worst_overall)
    if failures:
        print("\nBELOW 4.5:1 -- the base opacity is too high:")
        for vlabel, selector, cr in failures:
            print("  %.2f:1  %s  (%s)" % (cr, selector, vlabel))
        print("\nNot tuned around silently. Lower --prism-bg-opacity and "
              "re-run, or decide the trade explicitly.")
        return 1
    print("every sampled region holds >= 4.5:1 against the prism's brightest "
          "frame at this opacity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
