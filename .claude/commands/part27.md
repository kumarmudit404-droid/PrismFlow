---
description: PrismFlow Part 27 - AIT window-length and sample-mode table plus data card (stops for user decision)
---

PART 27 - WINDOW LENGTH, SAMPLE MODE AND DATA CARD
==================================================

Scenarios: russellmitchell and santos (both must be processed by /part26). If santos is missing,
run the table for russellmitchell only and say clearly that santos is pending.
Do not ask the user anything during this Part; the only decision is made AFTER it.

READ FIRST: docs/CONTRACT.md, docs/part26-ait-loader.md. Use the prismflow-honest-reporting skill.

STANDING RULES: frozen modules untouched; no fabricated numbers; statistics only, NO training;
raw/processed data stay gitignored; branch part27-ait-window; commit per step; do not push.

YOU MAY CREATE ONLY
  experiments/ait_window/run_window_table.py
  experiments/ait_window/README.md
  results/ait_window/*.json, *.csv, *.md
  docs/part27-ait-datacard.md

STEPS
1. For each scenario x sample_mode {network, host} x window {30, 60, 300, 900} s, run the Part 26
   loader and report, in the same row:
   - total samples, attack samples, benign samples, not-labelled samples, attack fraction
     (counted over label_known only; labels == -1 is never read as benign)
   - CONTIGUOUS ATTACK-RUN LENGTHS: the maximal runs of consecutive attack windows on the time
     grid, as the run count plus the run lengths longest-first (e.g. santos 300 s network:
     3 runs, 664 / 2 / 1), and the same for benign runs. The run count, not the attack-window
     count, is how many independent attack episodes the row actually contains, and Part 28 needs
     it to state what its +/- does and does not cover.
   - DNSTEAL-FAMILY SHARE: attack windows whose only attack steps are dnsteal-family
     (dnsteal, dnsteal-received, dnsteal-dropped, attacker), as a count and as a fraction of
     attack windows, and the count of attack windows carrying any other step. This sits in the
     same row as the attack fraction so neither can be quoted without the other.
   - per-view missing fraction (view_mask False)
   - per-view mean events per window
   - distinct attack steps landing in at least one attack window
   - views identical across the two scenarios? (yes/no, list differences)
2. santos at 60 s and 30 s each need THEIR OWN SCAN. Part 26's santos sweep has no 60 s row
   because `rewindow` only re-bins a 300 s scan upward (docs/part26-ait-loader.md section 16), so
   call scan_scenario with window_seconds=60 (and 30) for santos. Do not report the 60 s santos
   row as absent, do not estimate it from the 300 s row, and do not copy russellmitchell's.
3. SMALLEST USABLE WINDOW: for each scenario x sample_mode, report the smallest window among the
   lengths run that leaves AT LEAST 30 benign AND AT LEAST 30 attack windows (label_known only),
   with the two counts that decide it; if no length qualifies, say "none of the lengths tried"
   and give the best row. This is a reported threshold, not a choice - the decision still stays
   with the user at the gate below.
4. Save results/ait_window/window_table.{json,csv,md}. Every column above appears in all three,
   and each row carries its scenario, mode and window.
5. Data card docs/part27-ait-datacard.md: source URL, licence CC BY-NC-SA 4.0 with the exact
   attribution text and the papers the Zenodo page asks to cite, scenarios, views, label method,
   label coverage, unlabelled sources, same_sensor_group pairs, known limitations. State what the
   label means - "was the DNS exfiltration tool running in this window", not APT detection - and
   carry the dnsteal-family share and the attack-run counts into it.
6. Explain the trade-offs (short window = sparse views; long window = attack diluted and few
   samples; host mode = many missing views; fewer, longer attack runs = fewer independent attack
   episodes) but DO NOT choose.

STOP - REVIEW GATE
Print window_table.md, then end with exactly:
"PART 27 STOPPED: choose a window and sample mode, then run /part28 <window> <mode>."
