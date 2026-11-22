r"""Measure text contrast against the fixed BACKGROUND IMAGE. Real pixels.

    .venv\Scripts\python.exe web/check_hero_contrast.py

WHY THIS EXISTS, SEPARATELY FROM THE OTHER TWO
----------------------------------------------
``check_palette.py`` measures the PALETTE: token against token, and token
against the composited surfaces the palette itself defines.
``check_prism_contrast.py`` measures the animated WebGL prism, which needs a
GPU, a frame sweep and a stability control.

Neither covers the thing introduced on 2026-11-05: ``assets/img/
bg-spectrum-light.jpg`` is a photograph, its pixels are in no token file, and
since the backdrop became a fixed full-viewport layer it sits under EVERY
paragraph on the page rather than only the hero's. A palette can pass every
check in ``check_palette.py`` and still put body text on a rainbow. This
measures that and nothing else.

HOW IT MEASURES
---------------
1. The real site in the real browser, at a desktop and a phone viewport.
2. Text is made ``color: transparent``, not hidden, so each element keeps its
   own box and its own background and what is sampled is the pixels that would
   sit behind and between the glyphs.
3. The prism is hidden for the whole run. It is the other script's subject,
   and leaving it in would credit its pixels to the image.
4. Each region is captured twice, with ``.backdrop-bg`` visible and hidden. Only
   the pixels that DIFFER are the image's; everything else is the page's own
   composition, which ``check_palette.py`` already gates.
5. The crop is the element's CONTENT box, not its border box: each side is
   inset by that side's own border plus padding. Glyphs cannot be outside the
   content box, so nothing sampled is anything but background behind text.
   Without this, .limits-link failed twice on its own chrome and neither
   failure was about text -- first on the 1px semi-transparent border ring
   (3.43:1 over 6933 px), then, after a 2px inset, on the antialiasing of its
   999px-radius corner arcs (4.30:1 over 141 px). A border is a non-text UI
   component with a 3:1 floor and is drawn in a token check_palette.py gates.
6. The WORST-CONTRAST image pixel is kept, not the brightest and not the mean:
   a mean lets one bad line average away against the clean margin beside it.

EXIT CODE is 1 if any region falls below 4.5:1, so this can gate.
"""

from __future__ import annotations

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
except ImportError:  # pragma: no cover
    print("Pillow is required: .venv\\Scripts\\python.exe -m pip install pillow")
    raise SystemExit(2)

# The image is behind the whole page now, so this samples the whole page: the
# hero, which reads it through its own panel, and the chapters, which read it
# through the section scrim.
REGIONS = [
    (".hero__title", "the largest text on the site"),
    (".hero__thesis", "the quoted thesis, --text-secondary, the weakest token here"),
    (".hero .prose", "hero body copy"),
    (".hero .limits-link", "the one interactive element in the hero"),
    ("#ch1 .chapter__lede", "chapter opener, --text-secondary over the sheet"),
    ("#ch1 .prose p", "mid-page prose, the most common case"),
    ("#ch3 .prose p", "mid-page prose, further down"),
    ("#ch4 p", "prose immediately before the results views"),
    ("#limits .chapter__lede", "limitations hub opener"),
    (".site-nav__link", "a menu label over the frosted bar"),
    (".site-foot p", "the footer, --text-muted, the weakest text token on the page"),
]

VIEWPORTS = [(1600, 900, "desktop"), (390, 844, "narrow / phone width")]

PROBE_CSS = (
    "*,*::before,*::after{color:transparent !important;"
    "text-shadow:none !important;-webkit-text-fill-color:transparent !important;}"
    # Regions below the fold are scrolled to; a smooth scroll that has not
    # landed is read mid-flight and the crop points at the wrong pixels.
    "html{scroll-behavior:auto !important;}"
)

FLOOR = 4.5


def srgb_lum(rgb) -> float:
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def parse_rgb(css: str):
    nums = css.replace("rgba(", "").replace("rgb(", "").replace(")", "").split(",")
    return tuple(int(float(n)) for n in nums[:3])


def box_of(b, selector):
    """The element's CONTENT box in page coordinates."""
    return b.js(
        "(() => { const e = document.querySelector(" + json.dumps(selector) + ");"
        " if (!e) return null; const q = e.getBoundingClientRect();"
        " const c = getComputedStyle(e); const n = v => parseFloat(v) || 0;"
        " const l = n(c.borderLeftWidth) + n(c.paddingLeft);"
        " const rr = n(c.borderRightWidth) + n(c.paddingRight);"
        " const t = n(c.borderTopWidth) + n(c.paddingTop);"
        " const bm = n(c.borderBottomWidth) + n(c.paddingBottom);"
        " return JSON.stringify({x: q.left + scrollX + l, y: q.top + scrollY + t,"
        " w: Math.floor(q.width - l - rr), h: Math.floor(q.height - t - bm)}); })()")


def crop(b, r):
    data = b.ws.call("Page.captureScreenshot", {
        "format": "png", "captureBeyondViewport": True,
        "clip": {"x": r["x"], "y": r["y"], "width": r["w"], "height": r["h"],
                 "scale": 1}})["data"]
    return Image.open(_io.BytesIO(base64.b64decode(data))).convert("RGB")


def main() -> int:
    failures = []
    measured = 0
    b = Browser(gpu=True)
    try:
        for vw, vh, vlabel in VIEWPORTS:
            b.ws.call("Emulation.setDeviceMetricsOverride", {
                "width": vw, "height": vh, "deviceScaleFactor": 1,
                "mobile": vw < 700})
            b.goto(BASE + "/", settle=1.5)
            time.sleep(6.0)

            # Colours FIRST. Read after the probe and every one of them is
            # rgba(0,0,0,0), so every ratio would be computed against black.
            colours = {}
            for selector, _why in REGIONS:
                got = b.js("(() => { const e = document.querySelector("
                           + json.dumps(selector) + ");"
                           " return e ? getComputedStyle(e).color : null; })()")
                if got:
                    colours[selector] = got

            b.js("(() => { const s = document.createElement('style');"
                 " s.textContent = " + json.dumps(PROBE_CSS) + ";"
                 " document.head.appendChild(s); return 'ok'; })()")
            # The prism belongs to check_prism_contrast.py. visibility, not
            # display, so it cannot change layout between the two captures.
            b.js("document.getElementById('prism-bg').style.visibility = 'hidden'")
            time.sleep(0.6)

            print("\n" + "=" * 72)
            print("viewport %dx%d -- %s" % (vw, vh, vlabel))
            print("%-24s %-9s %-10s %-9s %s"
                  % ("region", "worst", "text", "px", "verdict"))

            for selector, why in REGIONS:
                # Scrolled to, then measured. scroll-behavior is forced to
                # auto by the probe CSS, so the jump has landed by the time
                # the rect is read.
                b.js("(() => { const e = document.querySelector("
                     + json.dumps(selector) + ");"
                     " if (e) e.scrollIntoView({block: 'center'});"
                     " return 'ok'; })()")
                time.sleep(0.45)
                raw = box_of(b, selector)
                if not raw or selector not in colours:
                    print("%-24s %s" % (selector, "not present -- not measured"))
                    continue
                r = json.loads(raw)
                if r["w"] < 2 or r["h"] < 2:
                    print("%-24s %s" % (selector, "zero-size box -- not measured"))
                    continue

                text_rgb = parse_rgb(colours[selector])
                text_lum = srgb_lum(text_rgb)

                on = crop(b, r)
                b.js("document.querySelector('.backdrop-bg').style.visibility = 'hidden'")
                time.sleep(0.4)
                off = crop(b, r)
                b.js("document.querySelector('.backdrop-bg').style.visibility = ''")
                time.sleep(0.4)

                on_px, off_px = list(on.getdata()), list(off.getdata())
                if len(on_px) != len(off_px):
                    print("%-24s %s" % (selector, "crop moved -- not measured"))
                    continue

                worst, worst_px, touched = None, None, 0
                for i, px in enumerate(on_px):
                    q = off_px[i]
                    # 2/255 per channel: above PNG rounding, below visible.
                    if abs(px[0] - q[0]) + abs(px[1] - q[1]) + abs(px[2] - q[2]) < 6:
                        continue
                    touched += 1
                    cr = ratio(text_lum, srgb_lum(px))
                    if worst is None or cr < worst:
                        worst, worst_px = cr, px

                if worst is None:
                    print("%-24s %s" % (selector,
                                        "image changes no pixel in this box"))
                    continue

                measured += 1
                ok = worst >= FLOOR
                print("%-24s %-9.2f %-10s %-9s %s"
                      % (selector, worst, "#%02X%02X%02X" % text_rgb,
                         "%d px" % touched,
                         "pass" if ok else "FAIL (< %.1f:1)" % FLOOR))
                if not ok:
                    failures.append((worst, selector, vlabel,
                                     "#%02X%02X%02X" % tuple(worst_px)))
    finally:
        b.kill()

    print("\n" + "=" * 72)
    print("regions measured: %d" % measured)
    if not measured:
        print("NOTHING WAS MEASURED -- that is a failure of this script, not a pass")
        return 1
    if failures:
        print("\nBELOW %.1f:1 -- text sits on the image where the image is too"
              " strong:" % FLOOR)
        for cr, sel, vp, px in sorted(failures):
            print("  %.2f:1  %s  (%s)  worst pixel %s" % (cr, sel, vp, px))
        print("\nFix the element, not the image: give it an opaque fill, move"
              " it clear, or change the token it is drawn in.")
        return 1
    print("every hero region holds %.1f:1 against the background image" % FLOOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
