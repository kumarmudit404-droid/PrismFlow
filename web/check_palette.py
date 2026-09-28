"""Measure a PrismFlow token palette. No dependencies.

    python web/check_palette.py                         # the live tokens.css
    python web/check_palette.py web/assets/css/x.css    # any token file

WHY THIS FILE EXISTS
--------------------
Phase (a) reported measured contrast, a luminance staircase and a CVD
separation for the palette, but the script that produced those numbers was
never committed, so none of them could be reproduced from the repository. This
is that script, written back. It reads the same tokens the site reads, so a
palette cannot pass here and fail in the browser.

Because the original script is gone, numbers from this one are NOT directly
comparable to the phase (a) commit message: a "separation of 8.3" means
whatever that script measured. This file states its own units -- CIEDE2000 in
CIELAB, and Machado 2009 CVD matrices at full severity -- and every palette is
measured with it, so comparisons between palettes here are like for like.

WHAT IT CHECKS
--------------
  1. every angle colour >= 3:1 against --base           (WCAG non-text minimum)
  2. every text token >= 4.5:1 against --base           (WCAG AA body text)
  3. accent pairs: whatever text sits on the accent >= 4.5:1
  4. the five angles form a MONOTONIC luminance staircase in spectral order,
     so they survive greyscale and print
  5. worst CIEDE2000 distance between any two angles under normal vision and
     under simulated deuteranopia, protanopia and tritanopia
  6. --status-not-measured is the lowest-chroma AND lowest-contrast token, so
     absent data can never read as data

Exit code is 1 if any hard check fails, so this can gate a commit.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# sRGB -> linear -> XYZ -> Lab
# --------------------------------------------------------------------------

def parse_hex(value: str) -> tuple:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def relative_luminance(rgb: tuple) -> float:
    r, g, b = (to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: tuple, b: tuple) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# D65, sRGB primaries.
_M = ((0.4124564, 0.3575761, 0.1804375),
      (0.2126729, 0.7151522, 0.0721750),
      (0.0193339, 0.1191920, 0.9503041))
_WHITE = (0.95047, 1.00000, 1.08883)


def to_xyz(rgb: tuple) -> tuple:
    r, g, b = (to_linear(c) for c in rgb)
    return tuple(m[0] * r + m[1] * g + m[2] * b for m in _M)


def to_lab(rgb: tuple) -> tuple:
    x, y, z = (c / w for c, w in zip(to_xyz(rgb), _WHITE))

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def chroma(rgb: tuple) -> float:
    _, a, b = to_lab(rgb)
    return math.hypot(a, b)


# --------------------------------------------------------------------------
# CIEDE2000
# --------------------------------------------------------------------------

def ciede2000(rgb1: tuple, rgb2: tuple) -> float:
    """Perceptual distance. ~1.0 is a just-noticeable difference."""
    l1, a1, b1 = to_lab(rgb1)
    l2, a2, b2 = to_lab(rgb2)

    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7))) if cbar else 0.5
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0

    dlp = l2 - l1
    dcp = c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2)

    lbar = (l1 + l2) / 2
    cbarp = (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbarp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbarp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbarp = (h1p + h2p + 360) / 2
    else:
        hbarp = (h1p + h2p - 360) / 2

    t = (1 - 0.17 * math.cos(math.radians(hbarp - 30))
         + 0.24 * math.cos(math.radians(2 * hbarp))
         + 0.32 * math.cos(math.radians(3 * hbarp + 6))
         - 0.20 * math.cos(math.radians(4 * hbarp - 63)))
    dtheta = 30 * math.exp(-(((hbarp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(cbarp ** 7 / (cbarp ** 7 + 25 ** 7)) if cbarp else 0.0
    sl = 1 + (0.015 * (lbar - 50) ** 2) / math.sqrt(20 + (lbar - 50) ** 2)
    sc = 1 + 0.045 * cbarp
    sh = 1 + 0.015 * cbarp * t
    rt = -math.sin(math.radians(2 * dtheta)) * rc

    return math.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2
                     + rt * (dcp / sc) * (dHp / sh))


# --------------------------------------------------------------------------
# CVD simulation -- Machado, Oliveira & Fernandes 2009, severity 1.0
# --------------------------------------------------------------------------

CVD = {
    "deuteranopia": ((0.367322, 0.860646, -0.227968),
                     (0.280085, 0.672501, 0.047413),
                     (-0.011820, 0.042940, 0.968881)),
    "protanopia":   ((0.152286, 1.052583, -0.204868),
                     (0.114503, 0.786281, 0.099216),
                     (-0.003882, -0.048116, 1.051998)),
    "tritanopia":   ((1.255528, -0.076749, -0.178779),
                     (-0.078411, 0.930809, 0.147602),
                     (0.004733, 0.691367, 0.303900)),
}


def simulate(rgb: tuple, kind: str) -> tuple:
    """Machado's matrices operate on LINEAR light, not on gamma-encoded bytes.

    Applying them to the 0-255 values -- which is what most snippets on the web
    do -- shifts every result and makes a palette look safer than it is.
    """
    m = CVD[kind]
    lin = [to_linear(c) for c in rgb]
    out = [sum(row[i] * lin[i] for i in range(3)) for row in m]
    return tuple(to_srgb(c) for c in out)


# --------------------------------------------------------------------------
# reading a token file
# --------------------------------------------------------------------------

TOKEN = re.compile(r"^\s*(--[a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{3,8})\s*;", re.M)

ANGLES = ["--angle-regulatory", "--angle-sentiment", "--angle-tech",
          "--angle-market", "--angle-financial"]
TEXT = ["--text-primary", "--text-secondary", "--text-muted"]


def read_tokens(path: Path) -> dict:
    return {m.group(1): m.group(2) for m in TOKEN.finditer(
        path.read_text(encoding="utf-8"))}


def report(path: Path) -> int:
    tok = read_tokens(path)
    missing = [k for k in ["--base"] + ANGLES + TEXT if k not in tok]
    if missing:
        print("MISSING TOKENS: %s" % ", ".join(missing))
        return 1

    base = parse_hex(tok["--base"])
    fails = []
    print("PALETTE: %s" % path.as_posix())
    print("  base %s  luminance %.4f\n" % (tok["--base"], relative_luminance(base)))

    # 1 + 2. contrast against the base
    print("  contrast against --base           value    floor")
    for key in TEXT:
        c = contrast(parse_hex(tok[key]), base)
        ok = c >= 4.5
        print("    %-22s %-8s %6.2f:1  4.5:1  %s"
              % (key, tok[key], c, "" if ok else "FAIL"))
        if not ok:
            fails.append("%s is %.2f:1, below 4.5:1" % (key, c))
    for key in ANGLES:
        c = contrast(parse_hex(tok[key]), base)
        ok = c >= 3.0
        print("    %-22s %-8s %6.2f:1  3.0:1  %s"
              % (key, tok[key], c, "" if ok else "FAIL"))
        if not ok:
            fails.append("%s is %.2f:1, below 3:1" % (key, c))

    # 3. text on the accent, if the palette declares the pair
    if "--accent" in tok and "--accent-ink" in tok:
        c = contrast(parse_hex(tok["--accent-ink"]), parse_hex(tok["--accent"]))
        ok = c >= 4.5
        print("\n  --accent-ink on --accent          %6.2f:1  4.5:1  %s"
              % (c, "" if ok else "FAIL"))
        if not ok:
            fails.append("--accent-ink on --accent is %.2f:1, below 4.5:1" % c)

    # 4. the luminance staircase, in spectral order
    print("\n  luminance staircase, spectral order")
    lums = [(k, relative_luminance(parse_hex(tok[k]))) for k in ANGLES]
    up = all(lums[i][1] < lums[i + 1][1] for i in range(len(lums) - 1))
    down = all(lums[i][1] > lums[i + 1][1] for i in range(len(lums) - 1))
    worst_step, worst_pair = None, None
    for i in range(len(lums) - 1):
        hi = max(lums[i][1], lums[i + 1][1]) + 0.05
        lo = min(lums[i][1], lums[i + 1][1]) + 0.05
        step = hi / lo
        if worst_step is None or step < worst_step:
            worst_step, worst_pair = step, (lums[i][0], lums[i + 1][0])
        print("    %-22s %.4f   step to next %.2f:1" % (lums[i][0], lums[i][1], step))
    print("    %-22s %.4f" % (lums[-1][0], lums[-1][1]))
    print("    monotonic: %s   worst adjacent step %.2f:1 (%s / %s)"
          % ("yes" if (up or down) else "NO", worst_step,
             worst_pair[0].replace("--angle-", ""),
             worst_pair[1].replace("--angle-", "")))
    if not (up or down):
        fails.append("the angle luminances are not monotonic in spectral order")

    # 5. separation, normal vision and simulated CVD
    print("\n  worst CIEDE2000 separation between any two angles")
    overall = None
    for kind in ["normal"] + list(CVD):
        worst, pair = None, None
        for i in range(len(ANGLES)):
            for j in range(i + 1, len(ANGLES)):
                a, b = parse_hex(tok[ANGLES[i]]), parse_hex(tok[ANGLES[j]])
                if kind != "normal":
                    a, b = simulate(a, kind), simulate(b, kind)
                d = ciede2000(a, b)
                if worst is None or d < worst:
                    worst, pair = d, (ANGLES[i], ANGLES[j])
        overall = worst if overall is None else min(overall, worst)
        print("    %-14s %6.2f   %s / %s" % (kind, worst,
              pair[0].replace("--angle-", ""), pair[1].replace("--angle-", "")))
    print("    worst across all four: %.2f" % overall)

    # 6. "not measured" must be the quietest token in the system
    nm = "--status-not-measured"
    if nm in tok:
        nm_rgb = parse_hex(tok[nm])
        nm_c, nm_chroma = contrast(nm_rgb, base), chroma(nm_rgb)
        # Compared against the DATA colours only. Body text is deliberately
        # near-neutral and therefore lower in chroma than this token; that is
        # not a violation, because text is not data. The rule is that no
        # colour used to encode a measured value may be quieter than the
        # colour that means "there is no value".
        louder = [(k, chroma(parse_hex(v))) for k, v in tok.items()
                  if k in ANGLES and chroma(parse_hex(v)) < nm_chroma]
        stronger = [k for k in ANGLES
                    if contrast(parse_hex(tok[k]), base) < nm_c]
        print("\n  %s  %s  contrast %.2f:1  chroma %.1f"
              % (nm, tok[nm], nm_c, nm_chroma))
        if louder:
            fails.append("%s is not the lowest-chroma token (%s below it)"
                         % (nm, ", ".join(k for k, _ in louder)))
            print("    FAIL lower chroma elsewhere: %s"
                  % ", ".join(k for k, _ in louder))
        if stronger:
            fails.append("%s is not the lowest-contrast token (%s below it)"
                         % (nm, ", ".join(stronger)))
            print("    FAIL lower contrast elsewhere: %s" % ", ".join(stronger))
        if not louder and not stronger:
            print("    the quietest token in the system, as required")

    print()
    if fails:
        print("  %d CHECK(S) FAILED" % len(fails))
        for f in fails:
            print("    - %s" % f)
        return 1
    print("  all checks pass")
    return 0


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or [HERE / "assets/css/tokens.css"]
    rc = 0
    for p in paths:
        if not p.exists():
            print("no such file: %s" % p)
            return 2
        rc |= report(p)
        print("-" * 72)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
