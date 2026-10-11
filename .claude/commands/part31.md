---
description: PrismFlow Part 31 - add real-data results to the existing web/ site (same theme)
---

PART 31 - REAL-DATA SECTION ON THE EXISTING WEBSITE
===================================================

READ FIRST: web/README.md, web/build_data.py (provenance(), clean(), every build_* function),
web/index.html, web/assets/js/charts.js, scores.js, limits.js, main.js, web/verify_page.py,
web/build_dist.py, and commit 45cb57c (the "needs more data" label pattern).
Use the prismflow-honest-reporting skill.

STANDING RULES
- Read-only over results/. No change to prismflow/, experiments/, results/, app.py.
- Same theme, palette, fonts and components as the current site. Run web/check_palette.py and
  web/check_prism_contrast.py; both must pass. No new colour scheme.
- Every number on the page comes from build_data.py reading a committed results/ file, with
  provenance (path + sha256) like the existing builders. A missing results folder renders the
  existing "needs more data" label with its cited reason - never a placeholder number, never 0.
- AIT-derived content shows the CC BY-NC-SA 4.0 attribution.
- Branch part31-web-real; commit per step; do not push.

YOU MAY MODIFY / CREATE ONLY
  web/build_data.py              (add builders; do not change existing ones)
  web/data/real_ait.json, web/data/real_lanl.json, web/data/real_summary.json
  web/assets/js/real_data.js     (new)
  web/index.html                 (add one section + nav link; existing sections unchanged)
  web/assets/css/*               (only new classes if unavoidable)
  web/verify_page.py             (add checks for the new section)

BUILD
1. build_real_ait() reads results/ait_clean/ and results/ait_attack/;
   build_real_lanl() reads results/lanl/; build_real_summary() builds the claim table.
2. New section "Real-data validation":
   - dataset cards: name, source link, licence, views, label method, scenario/days used
   - dependence heatmap per dataset (reuse the existing heatmap component)
   - detection table: ENIV clean vs attacked, detector AUC by k, mean +/- std, n seeds
   - honest-duplication result next to the synthetic clone_k2 figure
   - claim table: rows = detection, defence, honest-duplication false positive;
     columns = synthetic, AIT, LANL; cells = CONFIRMS / CONTRADICTS / NO EVIDENCE / needs more data,
     each linked to its results file
3. Mobile width 375px: no horizontal scroll.

CHECKS
- run build_data.py twice: identical JSON bytes
- web/verify_page.py passes; serve locally with web/serve.py and confirm no console errors
- save screenshots to web/shots/part31_*.png (desktop and 375px)

STOP - REVIEW GATE
End with: "PART 31 STOPPED: review web/shots/part31_*.png before /part32."
