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
4. **No CDN and no runtime download.** three.js is vendored under
   `assets/vendor/`; see `assets/vendor/README.md` for versions and licences.
   The only binary assets are `assets/img/nav-mark.png` and the hero background
   `assets/img/bg-spectrum-light.{jpg,webp}`, both committed files.

   **The typefaces are a system stack, not vendored files.** Inter and
   JetBrains Mono were asked for as local `@font-face` files. No font binary
   exists in this repository and nothing may be fetched at runtime or at build
   time, so there was nothing to vendor and nothing was downloaded. Both names
   lead their stack in `--font-sans` and `--font-mono`, so a reader who has
   them installed gets them; everyone else gets the platform UI face. This is
   the brief's own stated fallback, not a silent substitution.

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

**Spectrum light** (applied 2026-11-05): a near-white base, a violet UI accent,
a graphics-only six-stop spectrum, and the five categorical angle colours in
spectral order. It replaces variant B, "dark ember", which was the applied
theme from phase (f) step 1 until this change. `theme_preview.html` still shows
the two historical candidates and carries a banner saying its captions are the
old record.

The angle colours were **not re-searched**: they are the ones
`assets/css/tokens-a-light.css` had already searched for a light base, carried
across unchanged except `--angle-financial`, which was darkened from `#B58B00`
to `#B08700` because `#F7F8FA` is cooler than that file's `#FFFCF8` and the
original measured 2.97:1 against a 3:1 floor.

Two token names exist for several colours. `check_palette.py` reads literal hex
off `--base`, `--text-*`, `--accent*`, `--angle-*` and `--status-not-measured`,
and `app.css` plus every module in `assets/js` already consume those names, so
the plainer `--bg`, `--text-strong` and `--text` are `var()` aliases onto them
rather than a second set of literals. There is still one place each colour is
written down.

`--spec-1` .. `--spec-6` are **graphics only** — the page frame, the
scroll-progress line, the prism's rim. Measured on the base they are 2.61,
1.65, 1.20, 1.51, 1.61 and 3.09:1, so only `--spec-6` clears even the 3:1
non-text floor. They never carry text and never encode a value on their own.
The five `--angle-*` tokens remain the data palette and are gated.

**Colour lives in `assets/css/tokens.css` and nowhere else.** That was
aspirational until phase (f), which found 13 colour literals in `app.css` and 2
in `index.html` — one of them the page background itself. All are tokens now,
and `python web/check_palette.py` gates the result.

The accent is **UI chrome only**: surfaces, nav, buttons, links, focus rings. A
data colour is never the accent, so a measured value can never be mistaken for
a control.

Measured by `python web/check_palette.py` on the applied palette:

| | |
|---|---|
| angle colours vs base | all ≥ 3:1 (worst 3.13, financial) — **gated** |
| body text vs base | all ≥ 4.5:1 (worst 5.60, `--text-muted`) — **gated** |
| `--accent-ink` on `--accent` | 6.76:1 — **gated** |
| text on each composited surface | all ≥ 4.5:1 (worst 4.62, `--status-genuine-zero` on `--track`) — **gated** |
| luminance staircase, spectral order | monotonic — **gated** |
| worst adjacent luminance step | 1.35:1 (market / financial) — reported |
| worst simulated-CVD separation | 9.55 CIEDE2000 (protanopia, market / financial), across deuteranopia, protanopia and tritanopia — reported |
| `--status-not-measured` | the lowest-chroma, lowest-contrast token in the system — **gated** |

The composited-surface row is new in this theme and is check 7 in
`check_palette.py`. `--base` is the *easiest* background on the page; half the
palette's surfaces carry their own alpha and sit below it, so a token that
clears 4.5:1 on `--base` can still fail on `--track` or `--quote-bg`. Checking
only against `--base` would have shipped two such tokens: `--brand-orange` was
darkened from the requested `#B85E00` to `#9C5000`, and `--status-genuine-zero`
from `#6B7280` to `#5F6674`, so both clear the floor on every surface.

The five angles form a monotonic **luminance staircase**, so they remain
distinguishable in greyscale, in print, and under colour-vision deficiency —
not merely on a good monitor. On a light base contrast comes from being dark,
so the staircase runs downward and regulatory — the angle with no connector —
is the *highest*-contrast angle; its "unbuilt" reading is carried entirely by
`--angle-unbuilt-filter`.

`--status-not-measured` is 2.63:1 and is **deliberately below the text floor**.
It cannot be raised to 4.5:1 while check 6 holds: nothing encoding a measured
value may be quieter than the colour meaning "there is no value", and the
quietest angle is 3.13:1. The dark palette had the same property at 1.87:1, so
this is slightly louder, not a new compromise. Every "not measured" also
carries its words; the colour is never the only encoder.

Status colours are **never the only encoder**. `provider error` sits close to
the Sentiment angle in greyscale, so every status also carries a text label, and
charts add a hatch.

### Which adapter the prism contrast figures were measured on (2026-10-01)

> **Historical.** This subsection and the next describe the dark palette with
> the prism as a full-page background. Both are kept because the adapter note
> still governs how any WebGL contrast figure here should be read. The applied
> design, and its own measurements, are in *Spectrum light* below;
> `--prism-bg-opacity` is no longer a legibility knob and `PEAK_LUM` no longer
> exists.

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

### The hero: video removed, text centred, contrast re-measured (2026-10-01)

> **Historical**, superseded by *Spectrum light* below: the hero is neither
> text-only nor centred any more, and the prism is not a page background.

For one phase the hero held a looping video of the prism. **It was removed for
visual reasons**, along with its poster, its guard script and its three media
files. The hero is text only: title, thesis, prose, limits link — the same
words, unchanged.

With the video gone the hero became three stacked text blocks, so **the text is
centred**, the thesis' left border bar was replaced by a short centred rule in
`--accent`, and the hero's bottom padding dropped one step on the spacing scale
to close the space the video had occupied. Measured: the hero is 745px tall
where it was 1302px, and the gap to the Chapter 1 label is 132px where it was
156px.

Removing the video also removed the reason the hero carried `.no-prism-bg`, so
**the fixed background prism now shows behind the hero text**. That is a
contrast question, and it was measured rather than assumed — the hero title had
never been sampled before, because until now there was no prism beneath it:

| region | committed `6ae9645` | now, 1440px | now, 390px |
|---|---|---|---|
| hero title | not sampled | 10.40 | **not measured** |
| hero thesis | 4.94 | 4.89 | 4.88 |
| hero prose | 9.23 | 9.23 | 9.11 |

Worst anywhere on the page is unchanged at 4.64:1. Every sampled hero region
holds the 4.5:1 floor, so `--prism-scrim-local` was **not** applied and
`.no-prism-bg` was **not** restored; `--prism-bg-opacity` stayed at 0.40 and
`PEAK_LUM` at 0.52.

Two honest limits on that table. The hero title at 390px is **not measured**,
not a pass — at phone width the title does not overlap the prism and the probe
reports no changed pixel to sample. And these figures came from the headless
harness on *Microsoft Basic Render Driver* (WARP), so per the note above they
are **not like-for-like with the committed Intel UHD figures**. The thinnest
margin is the thesis at 4.88–4.89 against a 4.5 floor; that is the number most
worth re-checking on the real adapter.

### Spectrum light: the hero image and the re-anchored prism (2026-11-05)

> **Partly superseded** by *The fixed backdrop* below, which moved the image
> and the prism out of the hero and back into a fixed site-wide layer. The
> palette, the type scale and the contrast method in this section still
> stand; the hero-local geometry and the three-sided frame do not. Figures
> here that name an uncropped 1600x900 image predate the frame crop.

The hero now carries a committed background image,
`assets/img/bg-spectrum-light.jpg` (1600x900, with a `.webp` of the same pixels
at quality 82 offered first through `<picture>`). It is an `<img>`, not a CSS
background, specifically so the live prism can be positioned against the same
cropped box the image is cropped into.

**The anchor is measured, not eyeballed.** Scanning the JPG column by column
for saturated pixels, the painted rainbow fan collapses to a single row at
x = 0.72-0.73 and y = 0.562. The brief specified "~73% width, ~55% height"; the
pixels say 56.2%, and one point of hero height is ~9px at 1600x900. The prism
is placed at `left: 73%; top: 56.2%` of `.hero`, with
`transform: translate(-50%, -50%)`.

**The crop is pinned to that point, not centred.** `.hero-bg` carries
`object-position: 73% 56.2%`, the same pair the prism is placed at. With
`object-fit: cover`, an object-position of X% Y% puts the IMAGE's X%/Y% point
on the BOX's X%/Y% point, so the rays meet the glass at any box shape rather
than only where the box happens to match the image's 16:9.

Alignment, measured in the browser. "Offset" is the distance in px from the
prism canvas's centre to where the image's convergence point lands:

| viewport | hero box | prism centre | convergence point | offset |
|---|---|---|---|---|
| 1600x900 (16:9) | 1585x604 | 1157, 408 | 1157, 408 | **0, 0** |
| 2520x1080 (21:9) | 2505x604 | 1829, 408 | 1829, 408 | **0, 0** |
| 1200x900 (4:3) | 1185x604 | 865, 407 | 865, 407 | **0, 0** |
| 1024x768 | 1009x604 | 737, 464 | 737, 464 | **0, 0** |
| 800x900 | 785x604 | 573, 508 | 573, 508 | **0, 0** |
| 769x900 | 754x604 | 550, 508 | 550, 508 | **0, 0** |
| 390x844 (phone) | 390x914 | 265, 994 | 285, 724 | see below |

**`aspect-ratio: 16 / 9` was tried first and removed, because it was a bug.**
The idea was to stop `object-fit: cover` cropping at all. What it actually did:
at any viewport under about 1074px the `min-height` won on height, and the
ratio then drove the WIDTH from that height — so the hero came out 1074px wide
inside a 769px window. `overflow-x: clip` on the body hid the scrollbar, so the
only visible symptom was the prism sitting off the right edge of the screen.
`object-position` does the job properly and the hero is now plain
`width: 100%` with a `min-height` floor.

**On a phone the prism is deliberately not aligned to the rays.** The brief's
rule was "below 768px drop to 0.7 scale and move to 68%/50% so it clears the
text column". 68%/50% does *not* clear it: at phone width the hero text is
full-width and centred, so a prism at 50% of the hero's height falls inside
the copy's box at any horizontal position. MEASURED at 390x844, with the rest
of the mobile layout as shipped: restoring `68% / 50%` and the
briefed padding puts the prism's box at (200, 520)–(330, 650) in a 749px-tall
hero and gives **14689 px² of overlap with `.hero__thesis`** (0 px² against
every other hero element).

The clearance therefore has to be vertical. The hero gains bottom padding and
the prism sits in the band below the call to action, which measures 0 px² of
overlap against every hero text element.

The cost is that the prism is no longer on the convergence point at phone
width: the point is in frame — `object-position` pins it — but it lands behind
the body copy. Measured offset from the prism's centre to it at 390x844:
**-20, +270 px**. Text clearing the glass is worth more than a
rainbow meeting it. Measured box overlap with every hero text element, at six widths
from 390px to 2520px: **0 px² everywhere**.

The text column itself moved for the same reason: it was centred and
full-width, which put the last words of the thesis under the glass.
`check_prism_contrast.py` measured **1.11:1** there against
`--text-secondary`. No amount of tuning the prism fixes text under a
refracting solid, so the fix was to stop putting text there — above 768px the
column is left-aligned with a 4% gutter.

Its width is **solved against the prism, not chosen**. The prism's left edge
is at `73% - clamp(160px, 22vh, 340px)/2` of the hero, and that percentage
moves with the viewport because the prism is sized in vh and the hero in vw. A
flat 58% was tried and measured: clear from 1024px up, and 16px *into* the
thesis at 769px. The column is now
`min(46rem, calc(69% - clamp(160px, 22vh, 340px) * 0.5 - 2rem))`, which gives
a **38px gap at every desktop width from 769px to 2520px** and 0 px² of box
overlap with any hero text element at all seven widths tested.

#### What changed inside `hero3d.js`

The module was written for a near-black, full-viewport background. Four things
in it were actively wrong on a near-white hero, and each was found by looking
at a render rather than by reasoning about the code:

- **The five coloured beams are gone.** They are painted into the JPG now.
  Drawing them live as well put a second, differently-angled fan over the
  photograph's.
- **The bloom pass is gone**, with its `PEAK_LUM = 0.52` cap. The cap existed
  to hold the contrast floor for body text sitting over the prism, and no text
  sits over it any more. The bright-pass threshold of 0.55 was *below* the
  luminance of the backdrop plane's own white field, so bloom lifted the whole
  plane to 255,255,255 and drew it as a visible white card over the rays —
  four extra draws a frame to make the canvas worse.
- **Tone mapping is off.** ACES maps linear 1.0 to about 0.8, so the white
  backdrop, and the glass sampling it through transmission, both came out
  grey. The old bloom had been adding that 20% back by accident; with the
  bloom gone, the grey slab was all that was left.
- **The two radial pools behind the glass are gone.** On a dark page they gave
  the glass something bright to bend. On a white field the only thing a
  coloured pool can be is *darker* than its surroundings, and refracted
  through the prism the two of them came out as a pair of black eyes on the
  front face. The streaks stay: straight lines bend, and that is what reads as
  glass.

The backdrop plane itself stays, because it has to — three.js builds the
transmission target from the opaque objects only, and a transmissive solid with
nothing opaque behind it refracts the clear colour and renders as a grey slab.
It is unlit, 270x228 world units, and painted in `--prism-backdrop` `#FCFCFC`,
which is the pixel value the hero image actually has at the anchor (252,252,252
measured); at `--surface`'s pure white it showed as a faint card and at
`--base`'s `#F7F8FA` as a faint grey one. Measured in a screenshot, the plane
and the photograph beside it differ by 2/255.

Shadows were also switched off: nothing receives one now, and the contact
shadow under the glass is a CSS radial gradient. The material is the brief's —
`transmission: 0.95`, `roughness: 0.05`, `ior: 1.52`, tint `--prism-tint`
`#EDE9FF` — except `thickness`, where the brief's `1.2` is a unit-scale figure
and this scene works in SVG units with a 130-unit extrusion, so the ratio
carries over rather than the number. The rim cycles the six `--spec-*` tokens
over ~12s and is drawn, not additively blended: on a near-white page there is
no headroom left to add light to, and additive blending made the rim vanish.

**Reduced motion and no-WebGL are unchanged and still stronger than the
brief's ask.** The brief asked for "one static frame, no rotation" under
`prefers-reduced-motion`. `motion.js` never loads three.js at all in that case,
and never loads it without a real WebGL 2 context either, so both paths show
the static SVG prism at the same anchor and the 2.1MB module is never
requested. The SVG's `viewBox` was cropped from `0 0 900 240` to
`235 35 250 210` — the triangle plus margin — so the fallback frames the same
box the GL camera does, and its beams were removed along with the GL ones.

#### Type

The scale is the one specified — h1 `clamp(2.2rem, 5vw, 3.6rem)` at 1.08,
h2 `clamp(1.5rem, 2.6vw, 2.1rem)`, body `1.0625rem/1.65`, measure 68ch —
expressed on the existing `--step-*` tokens so no rule downstream changed.

**The rem bounds are divided by 1.18 and the vw terms are not.** `:root` sits
at `font-size: 118%`, so a rem here is 18.88px and the asked-for 2.2rem has to
be written `1.8644rem` to render at 35.2px. A vw is 1% of the viewport
whatever the root size is, so `5vw` stays `5vw`; scaling that too would have
made the h1 grow 18% faster than specified between its two bounds.

`.chapter__title` moved from `--step-3` to `--step-2`. Every `<h2>` on the
site carries that class, so while it sat on `--step-3` the h2 step the brief
specified was not reaching a single h2 — the titles rendered 28–40px against
the 24–34px asked for. `--step-3` has no users left and is deleted; the gap in
the numbering is deliberate, since renaming `--step-4` would churn every rule
that names it and change nothing on screen.

Figures carry `font-variant-numeric: tabular-nums`, set once on `body`. The
mono face was already applied per component to every class that holds a
number; this is the part that stops `1` being narrower than `8` inside it.

#### Contrast, re-measured for a photograph under the hero text

The hero background image is a thing no previous check covered.
`check_palette.py` measures tokens against tokens; `check_prism_contrast.py`
measures the WebGL prism. A palette can pass both and still put body text on a
rainbow, so **`web/check_hero_contrast.py` is new** and measures exactly that
and nothing else: the real site in the real browser, text made
`color: transparent` rather than hidden, the prism hidden for the whole run,
each region captured twice with `.hero-bg` shown and hidden, and only the
pixels that DIFFER scored — everything else is the page's own composition,
which `check_palette.py` already gates. The worst-contrast image pixel is
kept, never the mean.

| region | text token | desktop 1600x900 | phone 390x844 |
|---|---|---|---|
| `.hero__title` | `--text-primary` | 9.25:1 | 16.60:1 |
| `.hero__thesis` | `--text-secondary` | **5.39:1** | 9.62:1 |
| `.hero .prose` | `--text-primary` | 10.85:1 | 10.14:1 |
| `.hero .limits-link` | `--accent-text` | image changes no pixel | image changes no pixel |

The thinnest margin is the thesis at 5.39:1 against a 4.5 floor, on desktop,
where the image's rays pass behind the longest line.

`check_prism_contrast.py` now reports **0 regions scored, 18 examined and
clear** across both viewports: the prism changes no pixel inside any sampled
text box anywhere. That is the design — it is a hero element with the copy
held clear of it — and it is the reason the script had to learn the
difference between "examined and clear" and "never examined" before its exit
code meant anything. Before the scroll fix the same run examined only 8 of
the 18 pairs.

`.limits-link` reads "image changes no pixel" because it was **fixed, not
excused**. It had no fill — it did not need one over a plain base — and over
the image its `--accent-text` label measured **3.43:1 desktop / 3.68:1 phone**.
It now carries an opaque `--surface` fill, which covers its content box
completely, and the same text is 6.76:1 whatever is behind the pill.

Two measurement defects were found and fixed in the scripts rather than worked
around in the numbers:

- `check_prism_contrast.py` kept the **brightest** changed pixel as the worst
  case. That was right for a near-black page where every risk was a bright
  beam under pale text; on this palette the text is dark and the risk is the
  opposite, so "brightest" returned the *safest* pixel in the box. It now
  scores by the contrast ratio itself, which is correct on either palette.
- The same script read each region's crop box once, before hiding the prism,
  and reused it for every later capture. When the page moved in between — a
  smooth scroll still landing — the crops were of different pixels and the
  whole element counted as "changed". That produced a **1.93:1 FAIL over
  106245 pixels** for `header.hero p.prose` at 390px, where a direct diff of
  the same box shows the prism changing **zero** pixels and the prism's box
  sitting 108px below the text's. A moved region is now reported as not
  measured rather than scored.
- `check_hero_contrast.py` samples each element's **content box**, not its
  border box. Scoring the border box failed `.limits-link` twice on its own
  chrome and neither failure was about text: first its 1px semi-transparent
  border ring, then, after a flat 2px inset, the antialiasing of its
  999px-radius corner arcs.
- The first version of that moved-region guard compared the re-read box
  against the **inset** crop rather than the raw one, so it fired on every
  region by a constant 2px. Every region then reported "moved", nothing was
  scored anywhere, and the script printed *"every sampled region holds >=
  4.5:1"* and exited 0 — **a pass over zero measurements**, which is a worse
  failure than the false FAIL it replaced. Both halves are fixed: the guard
  compares like with like, and a run that scores nothing now says so and
  fails.
- `html { scroll-behavior: smooth }` is now overridden to `auto` inside the
  probe CSS. Every region is reached with `scrollIntoView`, which therefore
  ANIMATED; the script waits 1.2s, reads the rect, and on a long page the
  animation has not always landed, so the rect is read mid-flight and the
  region either clamps to a negative height or trips the moved-region guard.
  **Ten of eighteen region/viewport pairs came back unmeasured for that reason
  alone.** A jump and a glide end at the same scroll offset, so the page under
  test is identical in every other respect.
- Distinguishing the two reasons a region produces no number turned out to
  matter. "The probe worked and the prism touches no pixel in this box" is now
  the *expected* result almost everywhere, because the prism is a hero element
  and no body text sits on it; "the region was never examined" is a broken
  harness. They are counted separately — `regions scored` and `regions
  examined and clear` — and only the second being zero as well is a failure.

#### The frame, the progress line and the reveals

The spectrum border below the hero is CSS, not image: a masked pseudo-element
on `.frame`, inset 18px with a 20px bottom radius and a 2px gradient through
`--spec-1` .. `--spec-6`. `border-image` was not used because it ignores
`border-radius` and the corners would come out square; the top edge is removed
with `clip-path: inset(2px 0 0 0)`, which is safe because the element's top
corners are square by construction.

The reveals moved from an anime.js tween to a CSS class. `motion.js` now does
one thing — add `.is-visible` and set `--i` — and the 560ms
`cubic-bezier(.22, 1, .36, 1)`, the 20px offset and the 80ms stagger all live
in `app.css`. **The reveal-failsafe contract is unchanged**: content is visible
by default, CSS hides a target only while `<html data-reveal="armed">`, only
`reveal-failsafe.js` can set that attribute, and its 5s deadline clears it
whatever happens. anime.js is no longer imported by any page. The
scroll-progress line is a 2px gradient pinned under the sticky nav, driven by a
`--p` custom property written once per frame; with no JS `--p` is unset and the
line has zero width.

### The fixed backdrop, the reading sheet and the new menu bar (2026-11-05, second pass)

The spectrum image and the prism are **one fixed, full-viewport layer** again,
behind every pixel of content, with the whole page scrolling over it. The
previous pass had moved the prism into the hero; this restores the site-wide
behaviour it had before that, and puts the image in the same box so the two
cannot drift apart.

```
#backdrop          position: fixed; inset: 0; z-index: -1   image + prism + frame
  .backdrop-bg     object-fit: cover; object-position: 73% 56.8%
  #prism-bg        left: 73%; top: 56.8%, offset to its exit face
  ::after          the spectrum frame, inset 18px
.sheet             the measured scrim every chapter is read against
```

#### Three full-viewport layers were deleted, not re-layered

The backdrop image is an opaque photograph covering the viewport at z-index
-1. Everything that used to live beneath it had no visible pixel left:

- `#ambient`, a 2D canvas at z-index **-3**, was repainting five
  full-viewport radial gradients **every frame, forever**, under an opaque
  image. Its DPR cap, its frame-budget self-degrade and its blob-count
  fallback went with it. The drifting colour it provided is in the photograph.
- `body::before`, the fixed spectrum wash at -2.
- `body::after`, the fixed scrim at -2.

Nothing replaced them. This is the cheapest the page's background has ever
been: one image, one WebGL canvas, no per-frame 2D fill.

#### The scrim alpha is solved, not chosen

Body text scrolls over a moving prism again, so every chapter sits on
`--section-scrim`, one `.sheet` panel rather than a fill per section — the
sections are contiguous, so per-section fills would be the same sheet with a
seam at every boundary.

The solve: with the scrim removed, the prism driven across twelve frames of a
rotation and every text region sampled for its **darkest** composited pixel,

| region | text token | worst bare pixel | bare ratio | minimum alpha |
|---|---|---|---|---|
| `.chapter__lede` | `--text-secondary` | rgb(106,104,115) | 1.83:1 | **0.46** |
| `#limits .chapter__lede` | `--text-secondary` | rgb(107,105,116) | 1.86:1 | 0.46 |
| `#ch1 .prose p` | `--text-primary` | rgb(107,105,116) | 3.27:1 | 0.16 |
| `#ch4 p` | `--text-muted` | rgb(222,222,226) | 4.44:1 | 0.05 |
| hero title / prose | `--text-primary` | rgb(236,187,180) | 10.35:1 | 0.00 |

The binding case is a chapter lede in `--text-secondary` over the prism's own
edge lines. **0.46 is the floor; 0.65 is what shipped**, which measures
5.39:1 at that worst region and 10.86:1 for body prose while leaving 35% of
the backdrop showing. Two other rows of that sweep were discarded as
artefacts rather than acted on: the hero thesis's "worst pixel" was
rgb(91,63,212), its own `--accent` rule mark, and the nav brand's was
rgb(38,5,117), a pixel of the committed logo PNG. Neither is background
behind a glyph. `check_prism_contrast.py` diffs the prism in and out, so
neither reaches the gate.

#### The rays meet the glass: the anchor is the exit face, not the centre

Centring the prism's canvas on the convergence point left a visible gap. The
fan is a single faint line where it converges and only reads as rays once it
has spread, so the *visible* fan started clear of the glass.

`hero3d.js` works in the SVG's units: the triangle is `M360 40 L470 190 L250
190 Z` about a centroid of (360, 140) in a 250-unit frame, so for a square
canvas of side S the exit face's centre — world (55, 25) — sits **0.22S right
of and 0.10S above the canvas centre**. Offsetting the canvas by exactly that
puts the exit face on the ray origin.

0.22S also survives the rotation, which is the part that is not obvious. The
prism spins continuously about Y, so the silhouette sweeps; its *narrowest*
projected half-width is about 0.26S, half the 130-unit extrusion against the
250-unit frame. 0.22 is less than 0.26, so the ray origin stays inside the
glass at every phase rather than only at rest.

Measured, exit face against ray origin:

| viewport | prism size | exit face | ray origin | gap |
|---|---|---|---|---|
| 1600x900 (16:9) | 198px | 1157, 511 | 1157, 511 | **0, 0** |
| 2520x1080 (21:9) | 238px | 1829, 613 | 1829, 613 | **0, 0** |
| 1200x900 (4:3) | 198px | 865, 511 | 865, 511 | **0, 0** |
| 1024x768 | 169px | 737, 436 | 737, 436 | **0, 0** |
| 390x844 (phone) | 143px | 539, 909 | 539, 909 | **0, 0** |

The hero column is held off the glass by the same arithmetic:
`min(46rem, calc(69% - var(--prism-size) * 0.72 - 2rem))`. Writing that as
`73% - ...` and then pushing the column right by its 4% gutter put its right
edge 4% **past** the prism's left edge — measured, 1905 px2 of overlap at
1200x900 and 439 px2 at 1024x768. It is 0 px2 at all five viewports now.

#### One frame, and it belongs to the backdrop

The JPG carried a baked rounded frame, measured 13-24px in from each edge,
**and** `app.css` drew a CSS one at an 18px inset. Two rounded rectangles at
two different insets misalign the moment the image is cropped differently,
which is exactly what `object-fit: cover` does at every viewport. The baked
frame is **cropped out of the committed file** — 32px off each side, taking
the image from 1600x900 to 1536x836 — and the CSS frame is the only one.

Two things about it were wrong on first attempt and both were visible in a
render rather than deducible:

- Painted across its own box, the gradient put **one flat colour on each
  side**: the element is as tall as the document, around 60,000px here, so a
  viewport only ever sees a few hundred pixels of a 60,000px gradient. It is
  anchored to the viewport instead, so the visible frame always carries a
  spread of all six stops.
- Drawn as a fixed layer *above* the content it ruled a line **through every
  paragraph** crossing the top or bottom of the viewport. It is part of the
  fixed backdrop now, so the page scrolls inside it and it never touches a
  glyph. The `.sheet` is inset 8px further than the frame so the 2px line is
  never underneath it, dimmed to the sheet's own alpha.

#### The menu bar

Frosted glass over the moving backdrop: `--nav-bg` at `rgba(255,255,255,0.72)`
with `blur() saturate(1.6)`. The saturation matters — a plain blur over a pale
photograph turns the bar into flat grey, and it is the saturation that keeps
it reading as glass with colour moving behind it. `@supports not
(backdrop-filter)` falls back to **opaque** `--surface`, because without the
blur the backdrop would come through at full strength and take the labels'
contrast with it.

Three spectrum touches, all decoration and none of them text: a 2px hairline
along the top edge of the bar, a conic `--spec-*` ring around the PrismFlow
mark, and the scroll-progress line along the bottom. Every nav text state is a
gated token — `--text-secondary` at rest (9.98:1 on the glass fill),
`--text-primary` on hover (16.59:1), `--accent-text` for the limitations link
(6.76:1) and for the active chapter on `--accent-wash` (5.63:1).

The active state is a **wash, not the old solid `--accent` fill**: over a
moving backdrop a saturated block of colour in the bar competes with the
artwork, and the spectrum underline already says which chapter is current.
Focus is deliberately *not* the hover look — a keyboard user has to be able to
tell where focus is while the pointer hovers something else — so
`:focus-visible` is its own ring outside the pill.

**One row on a phone, scrolled rather than wrapped.** Nine items at 390px wrap
to three rows and take 210px of an 844px viewport — a quarter of the screen,
measured, before a word of the page is visible. The list is a single
horizontally scrollable rail below 768px and the brand word is hidden, leaving
the mark. It needs no `tabindex`: every child is a real link and focusing one
scrolls it into view by itself.

#### What the gates say now

**`check_prism_contrast.py`**, sixteen frames of a full rotation at two
viewports, the prism diffed in and out behind the measured scrim:

| region | text | desktop 1440x900 | phone 390x844 |
|---|---|---|---|
| `.hero__thesis` | `--text-secondary` | clear of the prism | **5.12:1** |
| `.hero p.prose` | `--text-primary` | clear of the prism | 9.00:1 |
| `#ch1 .chapter__lede` | `--text-secondary` | 5.14:1 | 5.12:1 |
| `#ch1 .prose p` | `--text-primary` | 9.10:1 | 9.00:1 |
| `#ch3 .prose p` | `--text-primary` | 9.10:1 | 9.00:1 |
| `#ch4 p` | `--text-muted` | 5.14:1 | 5.14:1 |
| `#limits .chapter__lede` | `--text-secondary` | 5.68:1 | 5.45:1 |

**12 regions scored, 6 examined and clear, worst 5.12:1.** The three hero
regions that read "clear" on desktop are clear by design: the hero column is
held off the glass there, so the prism genuinely touches no pixel of them. At
phone width the column is full-bleed, the prism is behind it, and the same
regions score. `.site-nav__brand` is clear at both, because the bar is at the
top of the viewport and the prism sits at 56.8% of it.

**A run that scores nothing is now a failure, not a pass.** While the prism
was a hero element, "examined and clear" everywhere was the right answer and
the only failure was never examining anything. With the prism behind every
paragraph there is no way for a healthy page to produce no scores, so the exit
condition was inverted to match the design.

The frame sweep was tightened at the same time. It used to be eleven samples
across 60s; it is now sixteen across 41.9s, which is exactly one revolution at
the prism's 0.15 rad/s, so every face the glass can present to the text falls
inside it. The two wobble terms (17.0s and 37.0s) move the pose by at most
0.06 and 0.05 rad, well under one 0.42 rad step, and are not separately swept.

**`check_hero_contrast.py`** was widened at the same time. The image is behind
every paragraph now rather than only the hero's, so its region list grew from
four hero blocks to ten covering the whole page, and the layer it hides moved
from `.hero-bg` to `.backdrop-bg`. 14 regions measured, **worst 5.56:1**
(`#ch4 p` in `--text-muted`, desktop); the hero title reaches 14.39:1 and body
prose 15.55:1. Regions reporting "image changes no pixel" are the ones sitting
on an opaque fill — `.limits-link` has its own `--surface` pill and the nav
labels sit on the frosted bar — which is the right answer, not a skipped test.

**`verify_page.py reveal`**: all 8 fault conditions pass, 115 targets, 0 left
hidden. Removing the ambient canvas did not touch the reveal guarantee, which
never depended on it.

**`check_palette.py`**: all checks pass, and `--section-scrim` was added to
the composited-surface check as its first row — it is the surface almost every
word on the site is actually read against. It composites at 5.65:1 for the
weakest text token over `--base`. That is the friendly case; the hostile case
is the backdrop's darkest pixel showing *through* it, and only
`check_prism_contrast.py` can see that.

## Motion, and the reveal guarantee

The scroll reveals hide real content so it can fade in. Hiding content is only
defensible if something guarantees it comes back, and that guarantee cannot live
in `motion.js`, because the failure it exists to survive *is* `motion.js` dying.
`assets/js/reveal-failsafe.js` is therefore a separate classic script, loaded
first in `<head>`: it owns the `data-reveal="armed"` attribute that CSS hides
against, and the 5s deadline that clears it. No un-hider loaded means no hiding,
so a missing or broken failsafe costs the fade, never the content.

The glass prism is optional and lazy. `three.js` (2.1 MB) is imported
dynamically only when a WebGL 2 context is actually obtained **and** motion is
not reduced. Any failure leaves the static SVG prism exactly where it is, at
the same anchor and framed to the same box. Phase (c) built this hero and
dropped it because it rendered as an opaque grey slab; phase (f) fixed the
cause — a transmissive material with no environment map and nothing opaque
behind it to refract — and restored it. The 2026-11-05 rework hit the same
grey slab twice more for two further causes, both recorded above and both at
the top of `assets/js/hero3d.js`.

Under `prefers-reduced-motion` nothing animates and three.js is never
requested at all — stronger than rendering one static WebGL frame, and it
costs the reader nothing to download.

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

That covers the palette against itself. Two further checks measure text
against things no token file describes, and both need the server running and a
real browser:

    .venv\Scripts\python.exe web/check_hero_contrast.py    # text over the background JPG
    .venv\Scripts\python.exe web/check_prism_contrast.py   # text over the live prism

`check_hero_contrast.py` takes a few minutes; `check_prism_contrast.py` sweeps
sixteen frames of one full prism rotation across two viewports and takes
roughly three quarters of an hour. The second one must SCORE regions: with the
prism a fixed backdrop behind every paragraph, a run that scores nothing means
the probe or the backdrop is broken, and it exits non-zero.

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
baseline, a throw right after `data-motion` is set, an `IntersectionObserver`
that never calls back, `motion.js` missing, `requestAnimationFrame` never
firing, no `IntersectionObserver` at all, `reveal-failsafe.js` missing, and
`prefers-reduced-motion: reduce` — and passes only if no reveal target is left
at computed opacity 0 after the deadline. The "never calls back" condition
replaced one that used to inject a failed anime.js import: once the reveals
stopped using anime.js that injection became a no-op, and a condition that
cannot fail is not a test. An observer that accepts every registration and
silently never fires is the real version of that failure, and is strictly
harder than deleting `IntersectionObserver`, because nothing can detect it. The key-leak scan must return zero
hits. `verify_page.py` also has `shot`, `eval`, `gpu` and `gpushot` for
screenshots and real-adapter checks; run it with no arguments for usage.

## Build phases, as built

- **(a) done** — tokens, chapter skeleton, real text pulled from the docs.
- **(b) done** — the data views wired to committed JSON only.
- **(c) done** — the scroll reveals, the ambient canvas, the reveal failsafe. The
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
