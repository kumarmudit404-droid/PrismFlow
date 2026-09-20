# Demo guide

```bash
streamlit run app.py --server.fileWatcherType none
```

The flag is required. See `docs/INSTALLATION.md` for why.

## A warning about every number on this screen

**The demo trains a small model on the spot, on one seed.** Nothing it displays
is evidence. The 5-SEED RULE applies to findings, and a single-seed live run is
not a finding no matter how clean it looks.

The numbers quoted in this guide are from an actual verified run on
2026-09-20 and are here so you know what a working app looks like. They are
illustrative. The project's real results are in `results/` and
`results/final_summary.json`.

## Layout

Three tabs:

1. **Independence budget** — the headline screen
2. **Run and results** — prediction, confidence, diagnostics
3. **Saved figures** — committed plots from `results/`, which *are* evidence

A **Scenario** selector and a **Run** button sit above them. Six scenarios:
`clean`, `clone`, `missing`, `noisy`, `chorus`, `adaptive`.

## Tab 1: the independence budget

This is the screen to show someone who has five minutes. It renders:

```
Nominal detectors:      4
Effective independent:  3.16
Independence efficiency: 79%

Most redundant pair: view 1 <-> view 2   (lambda_U = 0.16)
Mean cross-view dependence: rho_bar = 0.277

In plain language. You are paying for 4 independent sources of evidence.
You own about 3.2. Roughly 0.8 of them are ...
```

The framing is the point. An organisation that buys five detectors and runs
them on correlated inputs may be paying for five and owning two. That gap is
what ENIV measures, and it is the most legible thing this project produces.

Point at the plain-language line, not the heatmap.

## Tab 2: run and results

After pressing **Run**, on the `clean` scenario:

| panel | example value |
|---|---|
| Predicted class | 1 |
| Confidence (1 - vacuity) | 0.837 |
| Uncertainty (vacuity) | 0.163 |
| Top-label probability | 0.890 |
| Nominal views / ENIV | 4 / 3.16 |
| Efficiency ratio | 0.791 |
| lambda_U at q = 0.9 | 0.180 |
| Exceedances behind it | 25 |
| Mean unexplained agreement | +0.2862 |
| Flag rate | 5.1% |

Plus a per-view status panel (available / missing / noisy / discounted), a
dependence heatmap, and per-view evidence bars.

### The honest failure on screen

On that same clean run the app displays:

> **UNRELIABLE: 25 exceedances is below MIN_TAIL_SAMPLES = 50. Shown because
> hiding it would be worse, but do not read this number as evidence.**

This is the best thing in the demo and worth stopping on. The tail-dependence
estimate is underpowered at the demo's sample size, the code knows it, and it
says so instead of printing a confident number. That is `KNOWN_LIMITATIONS` L11
visible in the interface rather than buried in a document.

If you are demonstrating to a sceptical audience, show them this before you
show them anything that works.

## What to actually demonstrate

**The three-scenario arc, in this order.** Run each and watch the independence
budget and the confidence together.

| scenario | effective independent | efficiency | confidence | top-label prob |
|---|---|---|---|---|
| `clean` | 3.16 | 79% | 0.837 | 0.890 |
| `adaptive` | 2.95 | 74% | 0.598 | 0.565 |
| `chorus` | 2.70 | 67% | 0.599 | 0.535 |

The story that tells: under collusion the system does not merely lose accuracy,
it *notices* that its evidence has collapsed to fewer independent sources, and
its confidence falls accordingly — 0.837 to 0.599. A naive fusion would stay
confident, because three views still agree. Agreement is exactly what the
attack manufactures.

**Then say what it does not do.** The adaptive scenario is the one to be honest
about: Part 14 showed a single attack objective degrades both the discount and
the detection gate together (L8), and that the discount is partially evadable
(L12). The demo shows the defence working at one operating point. It is not
evidence that the defence holds.

## Tab 3: saved figures

Committed plots loaded from `results/`. Unlike tabs 1 and 2, **these are real
5-seed results.** Worth having open:

- `results/eniv_validation/estimator_vs_truth.png` — the estimator against
  analytic ground truth, the only validation of ENIV that exists
- `results/clone/confidence_vs_duplicates.png` — flat confidence under
  duplication, the founding result
- `results/tail/tail_vs_correlation.png` — the Part 13 negative result
- `results/robustness/missing_overview.png` — behaviour under missing views

## Scenario reference

| scenario | what it does | which limitation it illustrates |
|---|---|---|
| `clean` | no corruption | baseline |
| `clone` | duplicates a view exactly | L1 — flat confidence under duplication |
| `missing` | drops views | L5 — the detector inverts under missing evidence |
| `noisy` | adds per-view noise | L3 — the discount has no purchase on unreliable-but-independent views |
| `chorus` | the collusion attack | the attack the project was built around |
| `adaptive` | BPDA adversary that also suppresses measured dependence | L8, L12 |

## If something goes wrong

**App dies with `Tried to instantiate class '__path__._path'`** — the
`--server.fileWatcherType none` flag is missing.

**First run is slow** — it trains a model. Subsequent runs in the same session
reuse it.

**Numbers differ from this guide** — expected. Different machine, different
torch build, different draw. That is why none of them are evidence. If you need
numbers that hold, quote `results/final_summary.json`.
