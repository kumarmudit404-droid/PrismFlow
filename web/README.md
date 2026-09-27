# PrismFlow — Part 25 platform

One site holding both V1 and V2, as a narrative-first guided tour in five
chapters, with Known Limitations as a first-class section linked from every one.

    python web/serve.py      # http://127.0.0.1:825

## What this is, and what it is not

**Presentation only.** No experiment runs here, no metric is computed here, and
no pipeline code is touched. `app.py` is the V1 Streamlit app, frozen under
`v1-final`, and this directory does not import from it, modify it, or replace
it. The two can run side by side.

## Rules this site is built to

1. **Never display a number that is not in a committed file.** Every figure
   carries a `source:` line naming the file it came from. Where a figure does
   not exist, the site shows `not measured` — dimmed, dashed and labelled — and
   does not compute or estimate one. This is not decoration: V2-L6 is precisely
   that ECE and Brier *cannot* be computed yet, and a site that quietly filled
   those cells would be lying about the project's central result.
2. **No API calls, ever, and no credential in the bundle.** The site is static.
   `serve.py` sets a `connect-src 'self'` CSP so a stray external request fails
   loudly rather than silently becoming an undeclared dependency.
3. **The document root is `web/`, not the repository root** — `.env` holds real
   keys and must not be one URL away from a browser. Phase (b) copies the
   specific committed JSON the charts need into `web/data/`.

## Layout

    web/
      index.html            five chapters + the limitations hub
      assets/css/tokens.css palette and type scale -- the only place colour lives
      assets/css/app.css    layout and components
      assets/js/main.js     nav state + reduced-motion-aware reveal
      serve.py              loopback static server, web/ only
      data/                 (phase b) committed JSON copied in by the build step

## The palette

The repository contains **no logo**, and none was invented. The palette is
derived from the product's own thesis — one beam refracting into five angle
colours — and then measured:

| | |
|---|---|
| angle colours vs base | all ≥ 3:1 |
| body text vs base | all ≥ 4.5:1 |
| worst greyscale pair | 1.39 (target ≥ 1.25) |
| worst simulated-CVD distance | 8.3 (target ≥ 8), across deuteranopia, protanopia and tritanopia |

The five angles form a monotonic **luminance staircase** in spectral order, so
they remain distinguishable in greyscale, in print, and under colour-vision
deficiency — not merely on a good monitor. Regulatory is both the darkest and
rendered desaturated, because it is the angle with no connector: the fact is
visible before you read a word.

Status colours are **never the only encoder**. `provider error` sits at 1.03
greyscale ratio to the Sentiment angle, so every status also carries a text
label, and charts add a hatch.

## Build phases

- **(a) done** — tokens, chapter skeleton, real text pulled from the docs.
- **(b)** — charts and data views wired to committed JSON only.
- **(c)** — anime.js transitions and the single ray-traced hero.
- **(d)** — accessibility and performance pass, plus a key-leak grep of the
  build output for `sk-`, `gsk_` and the NewsAPI key pattern.
