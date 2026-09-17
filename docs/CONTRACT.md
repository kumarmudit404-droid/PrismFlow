# PrismFlow Contract

## 1. Project Statement

PrismFlow is a multi-view learning research project studying
Dependence-Discounted Evidential Fusion under Collusive View Compromise: it
tests the hypothesis that agreement between views should contribute evidence
in proportion to the effective independence of those views, not in
proportion to how many views agree, and that discounting fused confidence by
estimated inter-view dependence (ENIV) yields fusion that is robust to
colluding or compromised views.

**Evidence status for calibration (annotation, 2026-09-17).** The hypothesis
above is unchanged. What the evidence currently supports about calibration is
narrower than "PrismFlow improves reliability, not resolution," the framing
Part 07 started from. Under view duplication, PrismFlow preserves resolution,
accuracy and AURC relative to naive fusion. It does not show improved Brier
reliability or ECE, and its reliability trends slightly worse as copies are
added (5/5 seeds at k = 4). A naive model frozen at k = 0 rules out naive
fusion's own training adaptation as the explanation. The same frozen model
with the discount on at inference only (no retraining) reproduces PrismFlow's
robustness to added copies. So that robustness is a property of the discount
mechanism, at least for exact feature duplication. Training under the discount
removes a constant cost the untrained discount imposes even without duplicates.
Figures, confounds and scope are in `docs/EVALUATION_PROTOCOL.md`, "Current
claim". Any calibration claim must be stated consistently with that section.

## 2. Data Flow

```
views -> encoders -> evidence -> dependence -> ENIV -> discount -> fusion -> prediction + confidence + diagnostics
```

## 3. Component Classification

| Component            | Class                  | Notes                                      |
|-----------------------|------------------------|---------------------------------------------|
| Synthetic data gen     | ENGINEERING            | Deterministic given seed + config           |
| View encoders          | TRAINABLE NEURAL       | Per-view, no weight sharing by default      |
| Evidence heads         | TRAINABLE NEURAL       | Map encoder output to evidence [B, V, K]    |
| Dependence engine      | STATISTICAL ESTIMATOR  | No learnable parameters in V1               |
| ENIV                   | STATISTICAL ESTIMATOR  | No learnable parameters in V1               |
| Discount operator      | CLOSED-FORM OPERATOR   | Deterministic function of ENIV              |
| Fusion rule            | CLOSED-FORM OPERATOR   | Combines discounted evidence                |
| Calibration metrics    | STATISTICAL ESTIMATOR  | Computed post-hoc, no training               |
| Attack generators      | EXPERIMENT             | Produce compromised/colluding views          |
| Suspicion detector     | STATISTICAL ESTIMATOR  | Must not receive attack ground truth         |
| Training loop          | ENGINEERING            | Orchestrates trainable components            |
| Evaluation harness      | EXPERIMENT             | Multi-seed runs, aggregation, reporting      |
| Config / logging / seed / device / checkpoint utils | ENGINEERING | Shared infrastructure |

ENIV and the dependence engine are STATISTICAL ESTIMATORS with no learnable
parameters in V1. They must NOT be implemented as `nn.Module` subclasses.

## 4. Tensor Shape Conventions

| Tensor              | Shape        | Notes                          |
|---------------------|--------------|---------------------------------|
| `views`             | `[B, V, D]`  | Batch, view, feature dim        |
| `view_mask`         | `[B, V]`     | Boolean, True = view present    |
| `evidence`          | `[B, V, K]`  | Batch, view, class/evidence dim |
| `dependence matrix` | `[V, V]`     | Symmetric, per-batch or global  |

`B` = batch size, `V` = number of views, `D` = per-view feature dim,
`K` = number of evidence/classes.

## 5. Reproducibility Rules

- Every experiment has a unique experiment ID, a config file under
  `configs/`, and an explicit seed list.
- Every experiment writes all outputs to `results/<experiment_id>/`.
- Every experiment runs a minimum of 5 seeds and reports mean and standard
  deviation across seeds.
- Single-seed results are not evidence and must never be reported as a
  finding.

## 6. Rules for All Later Sessions

- Read `docs/CONTRACT.md` first, and only the files the current Part needs.
- Previously-passing modules are FROZEN. If you suspect a defect in an
  earlier module, REPORT it and stop. Do not fix it as a side effect.
- Never rename a public function or change a tensor shape silently.
- Never create or modify files outside the declared scope of the current
  Part.
- Never fabricate experimental results, not even as placeholders.
