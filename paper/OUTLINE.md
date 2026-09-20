# PrismFlow — paper outline

Target: 2 pages of outline for a full paper. Every figure below references a
file that exists under `results/`. Every number is a committed 5-seed result.

---

## Thesis (one sentence)

Agreement between views should contribute evidence in proportion to the
effective independence of those views, not in proportion to how many views
agree.

## The assumption under attack

**Dempster's rule of combination presumes independent evidence sources.**
Multi-view and multi-sensor fusion systems apply it — or an evidential
equivalent — to sources that are routinely correlated: different transforms of
one image, different models trained on one corpus, different detectors watching
one scene.

When sources are correlated, the rule counts the same evidence more than once.
Confidence rises with the *number* of agreeing views rather than with the
*amount of independent evidence* they carry. The failure is silent: accuracy
may be unchanged while confidence becomes unearned, and a confidence that is
unearned is worse than no confidence at all, because it is acted on.

The attack surface follows immediately. An adversary who cannot corrupt a view
convincingly can instead make several views *agree*, and be rewarded with
confidence rather than punished for it.

---

## Contributions

### 1. ENIV — the Effective Number of Independent Views

An estimator that maps a measured cross-view dependence matrix to an effective
independent view count in `[1, V]`, and a discount
`alpha_i = 1 / sum_j clip(R_ij, 0, 1)` that spends evidence at that rate rather
than at the nominal view count.

Dependence is measured on encoder features (not raw views), class-conditionally,
against a permutation null. The discount weights are deliberately detached from
the gradient.

*Validated:* correlation with analytic ground truth **0.9922**, MAE **0.5220
+/- 0.2998** effective views, at `n_views = 4` on synthetic data with
analytically known rho.

*Applied:* a standard 6-view benchmark (UCI `handwritten`) carries **3.28 +/-
0.07** effective views — an independence efficiency of **0.547 +/- 0.011**
against a permutation null of **5.51 +/- 0.08**.

This last measurement is the contribution that travels furthest. It needs ENIV
to be ordinally trustworthy, not numerically exact.

### 2. The Chorus attack

A collusion attack that perturbs `k` of `V` views toward mutual agreement on an
attacker-chosen target, rather than toward individual misclassification. It
exploits the assumption directly: it buys confidence with agreement.

Against an undefended evidential fusion it reaches **0.6267 +/- 0.1352** attack
success at `k = 3`, `eps = 2.0`. PGD on the same budget is the control.

### 3. The tail-dependence detector — **a pre-registered negative result**

*This contribution did not work, and the paper should say so in the abstract.*

The argument was that collusion should show up in the *upper tail* of the
cross-view evidence distribution more sharply than in mean correlation, since
the attack acts where views agree most strongly. A tail-dependence estimator
(`lambda_U`, empirical and copula-fitted) was built to test it.

**It is not sharper. It is worse.** At `eps = 2.0`, tail AUC is **0.5856 +/-
0.0126** against mean-dependence AUC **0.7085 +/- 0.1395** on identical data
and splits — near chance against a working detector. The tail flag is weakest
exactly where the batch-level signal is strongest.

The result is reported as a contribution because it was pre-registered as a
negative control, it closes a plausible line other groups would otherwise
spend effort on, and the mechanism is diagnosable (L7, L11).

---

## Figure list, in order

| # | figure | file | what it shows |
|---|---|---|---|
| 1 | ENIV against analytic truth | `results/eniv_validation/estimator_vs_truth.png` | The only validation of the estimator that exists. Correlation 0.9922; visible compression toward the range's middle. |
| 2 | Confidence under duplication | `results/clone/confidence_vs_duplicates.png` | The founding result: PrismFlow's confidence stays flat as exact copies are added; naive fusion's rises. |
| 3 | ENIV under duplication | `results/clone/eniv_vs_duplicates.png` | The mechanism behind figure 2 — the effective view count does not rise with the nominal one. |
| 4 | Calibration, clean data | `results/calibration/prismflow/reliability_diagram.png` | The discount does not cost calibration when there is nothing to discount. ECE 0.0637 +/- 0.0137. |
| 5 | Behaviour under missing views | `results/robustness/missing_overview.png` | Where the detector inverts (L5). Included because it is a failure. |
| 6 | Noise sensitivity | `results/robustness/noise_one_severe.png` | The scope boundary: the discount has no purchase on unreliable-but-independent views (L3). |
| 7 | Chorus attack, reliability | `results/comparison/chorus_k3/reliability_diagram.png` | Confidence under collusion at k = 3. |
| 8 | Tail dependence vs correlation | `results/tail/tail_vs_correlation.png` | Contribution 3's negative result. |
| 9 | Per-sample vs global discount | `results/per_sample/audit_gap.png` | The oracle-vs-deployable gap. |

Supporting tables (not figures): `results/final_summary.csv` (17 headline rows,
every value resolved from a committed file), `results/comparison/matrix.csv`
(the 9-condition detection matrix), `results/real/README.md` (the redundancy
audit).

---

## Limitations section

Full detail in `docs/KNOWN_LIMITATIONS.md` (L1-L15). The paper must carry at
least these seven:

1. **ENIV is validated only on synthetic data** where rho is a parameter we
   set, at a single design point (`n_views = 4`). Every real-data number is a
   measurement taken with an instrument calibrated elsewhere. (L9)
2. **The dependence estimator's bias is signed and exceeds its seed variance** —
   `+0.3345` at rho = 0, crossing zero near rho = 0.5, `-0.2203` at rho = 0.95,
   against an across-seed std of `0.0075`-`0.0235`. More seeds will not fix it.
   (L10)
3. **Tail dependence is unreliable below 50 exceedances**, and the project's own
   splits clear that floor by only 20%. (L11)
4. **The defence is partially evadable**, and the evading capabilities are
   specific: white-box with gradients, 3 of 4 views, `eps = 2.0`. A gate-aware
   attacker is **untested** — the arm built to test it is a null. (L12)
5. **The discount corrects inference but never shapes representation learning**
   (alpha is detached). A representation penalty moves dependence `0.3832 ->
   0.2052` where the discount cannot, but collapses the model at higher weights.
   (L13)
6. **One dataset family**, and its five seeds are five partitions of one
   dataset, not five datasets. (L14)
7. **No human-subject and no deployment evaluation.** Every claim is about
   measurements, not utility. (L15)

### The limitation that belongs in the abstract

**Detection and correction fail together.** One adaptive objective, containing
no gate term at any strength, simultaneously raises attack success (**+0.0333
+/- 0.0237**, 5/5 seeds), suppresses measured dependence (**-0.1057**, 5/5
seeds) and degrades the detection gate (**0.6743 -> 0.5806**, 5/5 seeds).
Pearson r between gate AUC and measured dependence across the sweep is
**0.9578**.

These were investigated as separate questions and reported in separate
documents. They are not separate defences: both are functions of cross-view
belief agreement. An adversarially robust successor needs a signal derived from
something else entirely — per-view reconstruction error, provenance metadata,
or out-of-distribution scoring on raw inputs — not a differently-weighted
combination of the same measurement. (L8)

---

## Suggested framing

The honest shape of this paper is **one positive measurement, one working
attack, and two negative results** — not a defence paper.

The strongest single claim is the redundancy audit: a widely used 6-view
benchmark carries roughly half the independent evidence its view count implies.
That is a statement about how multi-view benchmarks are built, it does not
depend on the defence holding, and it survives every limitation above except
L9 and L14.
