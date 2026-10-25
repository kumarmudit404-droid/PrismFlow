---
description: PrismFlow Part 37 - defence research in prismflow/v3 (L3, L4, L12, L5)
---

PART 37 - DEFENCE RESEARCH (V3)
===============================

READ FIRST: .claude/commands/limitations-index.md, docs/KNOWN_LIMITATIONS.md L3, L4, L5, L8, L12,
L13, experiments/chorus/ (Part 09), experiments/adaptive/README.md, experiments/synthesis/README.md,
results/chorus/attack_metrics.json. Use the prismflow-honest-reporting skill.

STANDING RULES: branch part37-v3-defence from main; commit per step; do not push. Frozen modules
import-only. Every new mechanism lives in a NEW package prismflow/v3/; nothing else under
prismflow/ changes. V1 results are references and are never re-labelled. 5 seeds, paired within
seed. Pre-register each hypothesis AND its falsification rule in
experiments/part37_v3/config.yaml and commit it BEFORE running anything. A result that fails its
rule is reported as a failure, in the same detail as a success.

YOU MAY CREATE ONLY
  prismflow/v3/** ; tests/unit/test_v3_*.py
  experiments/part37_v3/** ; results/part37_v3/**

A. L4 and L3: two discount variants in prismflow/v3/discount.py, each a drop-in for V1's
   evidence_discount: (i) CAPPED: each view's evidence contribution is limited to c times the
   median view's (c pre-registered, at least two values); (ii) PER-SAMPLE: the weight is computed
   per sample, not per batch. Evaluate on the committed results/chorus/perturbations/ tensors
   against naive fusion and V1 PrismFlow, per condition. This is a TRANSFER (non-adaptive)
   evaluation: the perturbations were optimised against V1, and the README says so.
   Pre-registered success rule: attack success below naive in at least 10 of 12 non-clean
   conditions, 5 of 5 seeds, with clean accuracy within 0.01 of naive. Anything less: L4 is
   CLOSED as confirmed for that variant.
B. L12: the gate-aware arm with epsilon lowered so the penalty is active, and a sweep over the
   number of compromised views k in {1, 2, 3}.
C. L5: an ADAPTIVE attacker against the best variant from A plus the detector, reusing Part 12's
   adaptive attacker by import. A variant that survives only the transfer evaluation in A is not
   called a defence.

BUDGET: time one seed per condition first; if a block projects past 90 minutes, run its smallest
pre-registered subset and mark the rest OPEN. L6, L7, L8 and L13 are out of scope unless A-C finish
early; if touched, pre-register first.

README: numbers, rule, verdict, proposed status per limitation; proposals to
docs/proposed_limitations_update.md. GATE: pytest -q, check_secrets.py --all, push scan; stop and
report on any match. Do not merge.
End with "PART 37 STOPPED: paste experiments/part37_v3/README.md into the review chat."
