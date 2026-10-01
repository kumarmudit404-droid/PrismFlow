# PrismFlow — Part 25 platform

One static site holding both V1 and V2, as a narrative-first guided tour in
five chapters, with Known Limitations as a first-class section linked from every
one, plus a separate Scores page that puts every committed number in one place.

    python web/build_data.py   # copy committed results into web/data/
    python web/serve.py        # http://127.0.0.1:825

`web/EDITING.md` is the map of the codebase: what each file does, which files
are generated, and the standing rules. Read it before editing anything here.

## What this is, and what it is not

**Presentation only.** No experiment runs here, no metric is computed here, and
no pipeline code is touched. `app.py` is the V1 Streamlit app, frozen under
`v1-final`, and this directory does not import from it at request time, modify
it, or replace it. The two can run side by side. The one place `app.py` is
imported is `web/build_v1.py`, an offline build step that runs the app's own
engine functions ahead of time so the site can show the app's real outputs
without training anything in a browser.

## Rules this site is built to

1. **Never display a number that is not in a committed file.** Every figure
   carries a `source:` line naming the file it came from. Where a figure does
   not exist, the site shows `not measured` — dimmed, dashed and labelled — and
   does not compute or estimate one. This is not decoration: V2-L6 is precisely
   that ECE and Brier *cannot* be computed yet, and a site that quietly filled
   those cells would be lying about the project's central result.
2. **No API calls, ever, and no credential in the bundle.** The site is static.
   `serve.py` sets `default-src 'self'; script-src 'self'; connect-src 'self'`
   (plus `img-src 'self' data:` and inline styles) so a stray external request
   fails loudly rather than silently becoming an undeclared dependency.
3. **The document root is `web/`, not the repository root** — `.env` holds real
   keys and must not be one URL away from a browser. `build_data.py` copies the
   specific committed JSON the charts need into `web/data/`, so "which files can
   this site read" is answered by listing one folder.
4. **No CDN and no runtime download.** anime.js and three.js are vendored under
   `assets/vendor/`; see `assets/vendor/README.md` for versions and licences.
   The display typeface is a system font stack, not a downloaded file.

## The five chapters, and what else is on the site

`index.html` carries all six sections in document order; the sticky nav links
each one.

| section | id | what it holds |
|---|---|---|
| Chapter 1 | `#ch1` | the problem, and the scope callout quoted from `docs/CONTRACT.md` |
| Chapter 2 | `#ch2` | V1 — what the synthetic experiments showed |
| Chapter 3 | `#ch3` | V2's challenge — real evidence, and the connector reality |
| Chapter 4 | `#ch4` | results, and what could not be measured |
| Chapter 5 | `#ch5` | interactive exploration (seven views, listed below) |
| Known limitations | `#limits` | the hub: V1's L1–L15 and V2's V2-L1 onwards, in full |

Chapter 5's views, in page order: the V1 engine run per scenario; the committed
V1 experiment figures; ENIV under duplication (k copies × ρ); attack success on
its own axes; every condition where PrismFlow lost to the baseline; the V2
evaluation dataset, all 48 rows; and per-row angle coverage over those rows.

`scores.html` is a second page sharing the same stylesheet, nav and mount-point
pattern. It holds KPI cards plus six filterable panels — V2 dataset composition
by domain, V2 per-angle retrieval coverage, real failures (V1 conditions
PrismFlow lost and V2 zero-claim rows), V1 attack success, V1 calibration under
duplication, and V1 ENIV under duplication. Filters are domain / angle / system;
a panel a filter does not apply to ignores it rather than guessing.

### The limitations hub is the only full copy

The entries are read from `docs/KNOWN_LIMITATIONS.md` and
`docs/v2-known-limitations.md` at build time, never retyped, so an entry
resolved in the document cannot stay open on the page. The chapters carry a
short status line and a link; the hub prints the entries.

### The dataset table

Chapter 5's dataset table is all 48 rows of `data/v2/evaluation_queries.json` —
the derived file the Part 24 harness actually loads — filterable, sortable and
searchable, with every ground-truth source as a link a reader can open and
check. The Part 24 workbook (`data/v2/part24_labeled_dataset.xlsx`) is offered
separately as a download so the two can be compared; both carry a sha256. The
table holds **no** prediction, score or calibration column, because none has
been computed on these rows. That is V2-L6. Row 040 is absent, and the footer
explains why in the words of the commit that left it out.

## The theme

Variant B, **"dark ember"**: a warm near-black base with an orange UI accent and
five categorical angle colours in spectral order. It was chosen in phase (f)
step 1 over a light variant, which is kept at `assets/css/tokens-a-light.css` so
the choice stays inspectable; `theme_preview.html` still shows both side by
side.

**Colour lives in `assets/css/tokens.css` and nowhere else.** That was
aspirational until phase (f), which found 13 colour literals in `app.css` and 2
in `index.html` — one of them the page background itself. All are tokens now,
and `python web/check_palette.py` gates the result.

Orange is **UI chrome only**: base, surfaces, nav, buttons, links, focus rings.
A data colour is never an orange, so a measured value can never be mistaken for
a control.

Measured by `python web/check_palette.py` on the applied palette:

| | |
|---|---|
| angle colours vs base | all ≥ 3:1 (worst 3.30, regulatory) — **gated** |
| body text vs base | all ≥ 4.5:1 (worst 5.01, `--text-muted`) — **gated** |
| `--accent-ink` on `--accent` | 7.20:1 — **gated** |
| luminance staircase, spectral order | monotonic — **gated** |
| worst adjacent luminance step | 1.32:1 (sentiment / tech) — reported |
| worst simulated-CVD separation | 11.01 CIEDE2000 (tritanopia, tech / market), across deuteranopia, protanopia and tritanopia — reported |
| `--status-not-measured` | the lowest-chroma, lowest-contrast token in the system — **gated** |

White on the bright orange accent fails at 2.60:1, which is why every orange
fill carries dark ink. The five angles form a monotonic **luminance staircase**,
so they remain distinguishable in greyscale, in print, and under colour-vision
deficiency — not merely on a good monitor. Regulatory is both the darkest and
rendered desaturated, because it is the angle with no connector: the fact is
visible before you read a word.

Status colours are **never the only encoder**. `provider error` sits close to
the Sentiment angle in greyscale, so every status also carries a text label, and
charts add a hatch.

### Which adapter the prism contrast figures were measured on (2026-10-01)

The table above is `check_palette.py` on the flat palette. A second, separate
measurement governs text sitting over the animated prism background:
`check_prism_contrast.py`, whose figures are recorded in commit `6ae9645`
(beam-peak cap `PEAK_LUM = 0.52`, `--prism-bg-opacity` held at 0.40, worst
sampled region 4.63:1). Those figures are GPU-dependent, because the quantity
measured is the luminance of a rendered WebGL frame. Which adapter produced
them is therefore part of the measurement, and this note records it.

- **The committed figures in `6ae9645` were measured on ANGLE / Intel UHD
  Graphics, hardware accelerated.** That is the run the 4.63:1 worst case and
  the whole before/after table come from.

- **The developer's normal Edge reports the same adapter.** Per an
  `edge://gpu` export taken 2026-10-01: WebGL hardware accelerated on Intel UHD
  Graphics, driver `32.0.101.5972`, display at 60 Hz. Recorded here as reported
  by that export; it was not produced by any harness in this repository.

- **The headless harness does not currently reach that adapter.**
  `verify_page.py`'s `Browser(gpu=True)` falls back to *Microsoft Basic Render
  Driver* (WARP), a software rasteriser, and reports it in the unmasked
  renderer string. A later contrast re-run was measured there. It is labelled
  **"consistent with no regression" and is NOT a replacement** for the
  committed table — a software rasteriser is not evidence about what the real
  adapter draws, which is the distinction `verify_page.py`'s own docstring
  already makes load-bearing. The committed `6ae9645` figures remain the only
  like-for-like contrast record.

- **Frame rate in the real browser: not measured.** The 60 Hz above is the
  display refresh rate the `edge://gpu` export states, not an achieved frame
  rate — the two are not the same number and should not be read as one. No
  harness in this repository has measured the site's actual frame rate on the
  real adapter. It stays **not measured** until a figure is supplied.

## Motion, and the reveal guarantee

The scroll reveals hide real content so it can fade in. Hiding content is only
defensible if something guarantees it comes back, and that guarantee cannot live
in `motion.js`, because the failure it exists to survive *is* `motion.js` dying.
`assets/js/reveal-failsafe.js` is therefore a separate classic script, loaded
first in `<head>`: it owns the `data-reveal="armed"` attribute that CSS hides
against, and the 5s deadline that clears it. No un-hider loaded means no hiding,
so a missing or broken failsafe costs the fade, never the content.

The glass prism hero is optional and lazy. `three.js` (2.1 MB) is imported
dynamically only when the hero is near the viewport **and** a WebGL 2 context is
actually obtained **and** motion is not reduced. Any failure leaves the static
SVG mark exactly where it is. Phase (c) built this hero and dropped it because
it rendered as an opaque grey slab; phase (f) fixed the cause — a transmissive
material with no environment map and nothing opaque behind it to refract — and
restored it. The whole explanation is at the top of `assets/js/hero3d.js`.

## What was dropped, and why

- **Liquid-glass buttons** (phase (f) step 3, a port of
  `kunal-chaudhary-design/liquid-buttons`). Built mid-session without
  authorisation and with none of the verification the hero got; verified on
  request, then **removed entirely** by the user's decision. Commit `21f3c95`
  reverted `motion.js`, `index.html`, `app.css` and `vendor/README.md` to their
  exact pre-feature content (zero diff) and deleted the two untracked files.
  Nothing on the site references `.liquid-btn`, `#liquid-stage` or
  `liquid_glass.js`. The site has exactly one WebGL feature: the hero prism.
- **The attack-success colour channel on the ENIV surface** — see below.
- **A logo.** The repository contains none and none was invented; the product's
  own prism *is* the mark.

## The join the brief asked for does not exist

The brief asked for the V1 surface "coloured by attack success". That colour
channel is not rendered, because no committed file supports it:

| | `results/clone_eigen/` | `results/chorus/` |
|---|---|---|
| rho | swept, {0.0, 0.5} | **fixed at 0.3**, not an axis |
| k | **duplicate copies** of view 0 | **colluding compromised** views |
| attack success | **absent entirely** | present |

The two `k` axes count different things and chorus's single rho is not one of
the grid's two values, so there is no cell anywhere in the repository giving an
attack success rate at a given (k, rho). Colouring the surface by it would have
invented a correspondence between two experiments. The surface therefore ships
with `attack_success: null` on all 30 cells, the page states why, and the real
chorus numbers are shown separately on their own axes — where they happen to
show the Part 09 finding directly: at eps 1.0, prismflow 0.481 ± 0.113 against
naive 0.455 ± 0.116.

## Why the ENIV column is empty

In the committed pass, **no row reached two angles**, so no row has an ENIV.
All 48 cells render `not measured`. That is the finding (V2-L6), not a gap in
the wiring — and it is exactly the case the "never show a guessed value" rule
exists for.

## How to edit this

**The five chapters** are hand-authored HTML in `web/index.html`, one
`<section class="chapter wrap" id="chN">` each, in document order, with the
limitations hub last. The Scores page is `web/scores.html`. Both are plain HTML:
every dynamic view is an empty mount point (`<div class="placeholder"
id="...">`) that a renderer in `assets/js/` fills. Change the copy in the HTML;
change a number by changing the committed result file it comes from and
re-running the build.

**Colour lives in `assets/css/tokens.css` only.** Nothing else hardcodes a
colour — not `app.css`, not `index.html`, not any `.js` file, which read tokens
at runtime through `getComputedStyle`. After any change there:

    python web/check_palette.py            # measures the applied palette
    python web/check_palette.py web/assets/css/tokens-a-light.css   # or a candidate

**After changing a committed result** under `results/` or `data/v2/`, re-run the
copy step, which re-stamps every sha256 and re-runs the credential patterns over
everything it emits:

    python web/build_data.py

Never hand-edit `web/data/*.json` — `build_data.py` overwrites it. If a V1
scenario output needs to change, re-run `python web/build_v1.py`, which drives
`app.py`'s own engine functions at a recorded seed.

**Before committing any change here**, start the server and run both checks:

    python web/serve.py                                   # in one terminal
    .venv\Scripts\python.exe web/verify_page.py reveal    # 8 conditions, expect 0 hidden

    # key-leak scan over everything the server can serve
    grep -rn -E "sk-[A-Za-z0-9]{8,}|gsk_[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_-]{20,}|\b[0-9a-f]{32}\b" \
      --include=*.html --include=*.js --include=*.css --include=*.json --include=*.py web/

The reveal matrix injects eight faults before the page's own scripts run —
baseline, a throw right after `data-motion` is set, a failed anime.js import,
`motion.js` missing, `requestAnimationFrame` never firing, no
`IntersectionObserver`, `reveal-failsafe.js` missing, and
`prefers-reduced-motion: reduce` — and passes only if no reveal target is left
at computed opacity 0 after the deadline. The key-leak scan must return zero
hits. `verify_page.py` also has `shot`, `eval`, `gpu` and `gpushot` for
screenshots and real-adapter checks; run it with no arguments for usage.

## Build phases, as built

- **(a) done** — tokens, chapter skeleton, real text pulled from the docs.
- **(b) done** — the data views wired to committed JSON only.
- **(c) done** — anime.js reveals, the ambient canvas, the reveal failsafe. The
  ray-traced hero was attempted here and dropped.
- **(d) done** — accessibility and performance pass, and the headless harness
  (`verify_page.py`) the later phases are verified with.
- **(e) done** — the V1 engine's own outputs, the committed figures, the real
  failures, the 48-row dataset table, and one copy of the limitations read from
  the documents.
- **(f) done** — the dark-ember theme with colour actually moved into
  `tokens.css`, the glass prism hero fixed and restored, the Scores page, and
  the liquid-glass buttons built, verified and then removed.

## Credit

The site was built with [Claude Code](https://claude.com/claude-code) as the
development tool, across the phases listed above. The commit history records
which phase and which model produced each change.
