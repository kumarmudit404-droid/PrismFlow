---
description: PrismFlow Part 33 - final report (every number cited to results/)
---

PART 33 - FINAL REPORT
======================

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md, docs/v2-known-limitations.md,
docs/PROJECT_STATUS.md, docs/EVALUATION_PROTOCOL.md, and every README under experiments/.
Use the prismflow-honest-reporting skill.

STANDING RULES: docs only - no code, results or web changes; every number cites its results/
file (and field); anything without a results file goes under "Not measured"; branch part33-report;
commit per step; do not push.

YOU MAY CREATE / MODIFY ONLY
  docs/FINAL_REPORT.md
  docs/PROJECT_STATUS.md (status lines only)
  docs/proposed_limitations_update.md (proposed edits; do NOT edit KNOWN_LIMITATIONS.md directly)

SECTIONS
1. Problem and thesis
2. Method (data flow from CONTRACT section 2, ENIV, soft-cluster discount, fusion)
3. Synthetic results (Parts 06-14)
4. Real-data results (Parts 15, 28-30): AIT and LANL
5. What works: detection - with numbers per dataset
6. What does not: defence - with the mechanism (L3, L4, L5, L8)
7. V2 status (V2-L1 to V2-L9)
8. Limitations and threats to validity (testbed vs real network, label leakage on LANL auth,
   window choice)
9. Future work: a signal that is not cross-view agreement (timestamps/provenance),
   Part 20 verification (blocked on API funding), Part 24 calibration (not measured)
10. Not measured - explicit list

DONE WHEN: a script check confirms every number in FINAL_REPORT.md appears in a cited results/
file; list any that do not and fix the text, not the results.
