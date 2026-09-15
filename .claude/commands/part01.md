---
description: PrismFlow Part 01
---

PART 01 - CONTRACT, SKELETON, CONFIGURATION
===========================================

You are the lead engineer for PrismFlow, a multi-view learning research project.

PROJECT NAME
PrismFlow - Dependence-Discounted Evidential Fusion under Collusive View Compromise.

CORE HYPOTHESIS
Agreement between views should contribute evidence in proportion to the effective
independence of those views, not in proportion to how many views agree.

SCOPE OF THIS PART ONLY
Create the contract document, package skeleton, and shared utilities.
Create NO machine learning code.

YOU MAY CREATE ONLY THESE PATHS
  CLAUDE.md
  docs/CONTRACT.md
  prismflow/__init__.py and empty __init__.py under:
      data/ models/ statistics/ eniv/ attacks/ evaluation/ utils/ app/
  prismflow/utils/config.py
  prismflow/utils/seed.py
  prismflow/utils/logging.py
  prismflow/utils/device.py
  prismflow/utils/checkpoint.py
  configs/default.yaml
  tests/unit/test_utils.py
  requirements.txt
  .gitignore
  README.md
  Empty dirs with .gitkeep: data/raw data/processed experiments results logs checkpoints

docs/CONTRACT.md MUST BE UNDER 150 LINES and must contain:

  1. One-paragraph project statement.

  2. Data flow on one line:
     views -> encoders -> evidence -> dependence -> ENIV -> discount ->
     fusion -> prediction + confidence + diagnostics

  3. A component table classifying EVERY component as exactly one of:
     TRAINABLE NEURAL / STATISTICAL ESTIMATOR / CLOSED-FORM OPERATOR /
     EXPERIMENT / ENGINEERING.
     State explicitly: ENIV and the dependence engine are STATISTICAL ESTIMATORS
     with no learnable parameters in V1. Do not make them nn.Module subclasses.

  4. Tensor shape conventions.
     views [B, V, D], view_mask [B, V] boolean, evidence [B, V, K],
     dependence matrix [V, V].

  5. Reproducibility rules. Every experiment has an ID, a config file, a seed
     list, and writes to results/<experiment_id>/.
     EVERY experiment runs a minimum of 5 seeds and reports mean and standard
     deviation. Single-seed results are not evidence and must never be reported
     as a finding.

  6. Rules for all later sessions:
     - read docs/CONTRACT.md first, and only the files the current Part needs
     - previously-passing modules are FROZEN. If you suspect a defect in an
       earlier module, REPORT it and stop. Do not fix it as a side effect.
     - never rename a public function or change a tensor shape silently
     - never create or modify files outside the declared scope of the Part
     - never fabricate experimental results, not even as placeholders

CLAUDE.md (project memory, auto-loaded each session) must be under 60 lines and
contain only: the one-sentence thesis, the data flow line, the FROZEN-modules
rule, the 5-seed rule, and the never-fabricate rule. Keep it short; it is
prepended to every session.

CONFIG REQUIREMENTS
  YAML-backed with dot access. No hard-coded paths anywhere in the codebase.
  Deterministic seeding across python / numpy / torch including CUDA.
  CPU/GPU auto-detection.

VERIFY
  Run pytest. Confirm the package imports cleanly.
  Confirm two runs with the same seed produce bit-identical random tensors.

STOP after tests pass. Do not implement encoders, evidence, ENIV, or fusion.
