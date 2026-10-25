# PrismFlow — final report

**Date:** 2026-10-25. **Branch:** `part33-report`. **Scope:** Part 01 to Part 32.

This report is a reading of committed files. Every number in it appears in the
file its section cites, and `docs/check_report_numbers.py` is what proves that:
it extracts each number, resolves it against the cited file, and fails on any
number it cannot find. Where no committed file holds a quantity, the report
says **not measured** and does not estimate one. Section 10 lists every such
absence in one place.

Where this report and `docs/KNOWN_LIMITATIONS.md`, `docs/v2-known-limitations.md`
or `docs/CONTRACT.md` disagree, those are authoritative and this is wrong.

---

## 1. Problem and thesis

Multi-view and multi-source systems raise confidence when their views agree.
Agreement is cheap when the views are not independent: two sensors reading the
same underlying signal, two retrievers returning the same document, two
compromised views coordinated by an attacker all produce agreement that looks
like corroboration and is not.

**Thesis.** Agreement between views should contribute evidence in proportion to
the *effective independence* of those views, not in proportion to how many
views agree.

The thesis splits into two claims that must be stated separately, because the
project's central result is that one holds and the other does not:

- **Detection.** Inter-view dependence is measurable, and collusion raises it.
- **Defence.** Discounting fused confidence by that dependence makes fusion
  robust to colluding or compromised views.

**Detection is evidenced on synthetic data. The defence is refuted on
synthetic data, and is not validated on either real dataset.** Sections 5 and
6 give the numbers; section 4 gives what the real data can and cannot say.

**Sources:** `docs/CONTRACT.md`, `docs/KNOWN_LIMITATIONS.md`.

---

## 2. Method

The data flow, from `docs/CONTRACT.md` section 2:

```
views -> encoders -> evidence -> dependence -> ENIV -> discount -> fusion -> prediction + confidence + diagnostics
```

**Dependence.** Per-pair dependence between views is estimated on encoder
features or on per-view evidence, conditioned on the predicted class so that
within-class correlation is what gets measured, with a chance correction
estimated by permutation.

**ENIV — the effective number of independent views.** The dependence matrix is
reduced to an effective view count, in two forms that are reported separately
and never averaged: an eigenvalue form (the headline) and a design-effect form.
A set of views with nominal count `V` and no dependence reads near `V`; exact
clones do not raise it.

**Soft-cluster discount.** Each view receives a weight `alpha` from a soft
clustering of the dependence matrix, and its evidence is scaled by that weight
before fusion. The discount is **proportional, not capping**, and it is
**detached**: it corrects inference and never shapes the representation.

**Fusion.** Discounted evidence becomes a subjective-logic opinion per view;
opinions are combined by Dempster's rule into a fused belief and uncertainty.

**Diagnostics.** Alongside the prediction, the system reports the dependence
matrix, ENIV, the per-view weights, and a suspicion score — unexplained
agreement, in cosine units — which is the detector half of the thesis.

Two properties of the method bound everything that follows. The discount is
proportional rather than capping and sublinear in `alpha`, and it is applied
per batch rather than per sample. The detector and the correction read the
*same* quantity — cross-view belief agreement — so when the signal is wrong,
both are wrong together.

**Sources:** `docs/CONTRACT.md`, `docs/FINAL_ARCHITECTURE.md`.

---

## 3. Synthetic results (Parts 05–15)

Synthetic data is the only place `rho` is a parameter that was *set*, so it is
the only place the estimator has a ground truth to be validated against. The
seed counts are not uniform. 3.1's correlation is a single pooled value with
no recorded seed count (`results/final_summary.json` rows[0] carries
`"std": null, "n_seeds": null`), so it is not a 5-seed figure and must not be
read as one. The two detector AUCs in 3.3 are 5 seeds each per
`results/comparison/detector.json` (`summary.clone_k2.auc.n_seeds` and
`summary.missing_50.auc.n_seeds`). Every other row below is 5 seeds, mean
+/- sample std.

### 3.1 The estimator reads what it is supposed to read

| quantity | value |
|---|---|
| ENIV correlation with the analytic effective view count | 0.9922 |
| ENIV mean absolute error | 0.5220 +/- 0.2998 effective views |

The estimator tracks analytic truth in **rank** but not in **value**: about
half an effective view of absolute error, at four nominal views only.

### 3.2 Clean behaviour

| quantity | value |
|---|---|
| accuracy on clean synthetic data | 0.83 +/- 0.0797 |
| expected calibration error on clean synthetic data | 0.0637 +/- 0.0137 |

### 3.3 Detection, and where it fails

| condition | detector AUC |
|---|---|
| exact clones (`clone_k2`) | 0.8743 +/- 0.0583 |
| half the views missing (`missing_50`) | 0.1247 +/- 0.0319 |

The clone figure is **not like-for-like**: `clone_k2` runs six views against a
four-view clean reference, and the suspicion score is a maximum over
off-diagonal pairs, which grows with width. Part of the 0.8743 is the width
change itself, and it must not be differenced against any fixed-width result
(L22). The missing-view figure is the detector **inverting** — well below
chance, not merely degraded (L5).

### 3.4 Tail dependence is not the sharper instrument

| detector, same data and split | AUC |
|---|---|
| tail dependence | 0.5856 +/- 0.0126 |
| mean dependence (`rho_bar`) | 0.7085 +/- 0.1395 |

The quantity predicted to be sharper is the weaker of the two (L7).

### 3.5 A representation penalty moves what the detached discount cannot

| system, same operating point | mean dependence |
|---|---|
| V2 orthogonality penalty | 0.2052 +/- 0.0550 |
| V1 baseline | 0.3832 +/- 0.0231 |

V1 detaches the discount, so it corrects inference but never shapes
representation (L13).

### 3.6 The adaptive attacker

| quantity | value |
|---|---|
| attack success, adaptive attacker | 0.6600 +/- 0.1161 |
| attack success, baseline chorus attack | 0.6267 +/- 0.1352 |
| deployable gate AUC under the adaptive attacker | 0.5806 +/- 0.0413 |
| oracle gate AUC, fitted with attack labels — an upper bound | 0.7448 +/- 0.0592 |

An attacker that is *told* to suppress measured dependence gains while doing
so, and the deployable gate degrades without being targeted (L8). The oracle
gate is fitted with attack labels no deployment has; the gap between it and
the deployable gate is the finding, and the oracle number is not a defence.

### 3.7 Applied to a standard benchmark

| quantity — Handwritten (UCI Multiple Features), six nominal views | value |
|---|---|
| measured ENIV | 3.28 +/- 0.07 |
| efficiency ratio | 0.547 +/- 0.011 |
| upper tail dependence at the 0.9 quantile | 0.127 +/- 0.042 |
| ENIV after within-class feature permutation — the independence control | 5.51 +/- 0.08 |
| deficit against that control | -2.23 +/- 0.06 |

A standard six-view benchmark carries about **3.28** effective views. The
permutation control makes the views independent by construction while leaving
sample size, feature dimension and marginals untouched, so the deficit against
it is measured redundancy and the remainder is estimator floor. **ENIV is
applied here, not validated here**: real views carry no ground-truth
dependence (L9).

**Sources:** `results/final_summary.json`, `results/comparison/detector.json`, `results/real/README.md`.

---

## 4. Real-data results (Parts 28–30)

Two real datasets. Neither is pooled with the other, and AIT's two directions
are never averaged.

### 4.1 AIT Log Data Set V2.0 — what the label actually is

The label is **"was the DNS exfiltration tool running in this window"**, not
"detect an APT" and not "detect a multi-step intrusion". The multi-step
intrusion is three windows on `santos` and four on `russellmitchell`; no
metric here is computed over those windows.

**Direction A** — train `russellmitchell`, test `santos`; 1108 label_known
test windows, 667 attack, 441 benign, prevalence 0.602; **three** independent
contiguous attack blocks:

| metric | naive fusion | V1 PrismFlow | majority baseline |
|---|---|---|---|
| balanced accuracy | 0.9444 +/- 0.0056 | 0.9487 +/- 0.0061 | 0.5000 |
| ROC AUC | 0.9965 +/- 0.0008 | 0.9962 +/- 0.0005 | 0.5000 |
| Brier | 0.0819 +/- 0.0092 | 0.0707 +/- 0.0071 | 0.4792 |
| ECE | 0.0314 +/- 0.0070 | 0.0194 +/- 0.0042 | 0.0000 |
| ENIV, eigen form | 3.4507 +/- 0.0790 | 3.4958 +/- 0.1015 | n/a |
| mean dependence | 0.2714 +/- 0.0154 | 0.2548 +/- 0.0244 | n/a |

**Direction B** — train `santos`, test `russellmitchell`; 1145 label_known
test windows, 1031 attack, 114 benign, prevalence 0.900; **one** independent
contiguous attack block:

| metric | naive fusion | V1 PrismFlow | majority baseline |
|---|---|---|---|
| balanced accuracy | 0.9500 +/- 0.0685 | 0.9648 +/- 0.0455 | 0.5000 |
| ROC AUC | 0.9999 +/- 0.0002 | 0.9999 +/- 0.0002 | 0.5000 |
| Brier | 0.2237 +/- 0.0768 | 0.1884 +/- 0.0688 | 0.1793 |
| ECE | 0.2066 +/- 0.0602 | 0.1921 +/- 0.0478 | 0.0000 |

The two directions must be read separately. Direction B reaches ROC AUC
0.9999 while its Brier and ECE are **worse than the majority baseline's**
0.1793 and 0.0000: discrimination and calibration come apart, and the positive
class is one contiguous block (L16). Every +/- is a spread over seeds, not
over independent attacks.

### 4.2 AIT is solved by one view, so it is not evidence about fusion

A logistic regression on the nine `dns` features alone scores balanced
accuracy **0.9898** and ROC AUC **0.9993** on `santos`, and **1.0000** on both
measures on `russellmitchell`. No other single view clears balanced accuracy
**0.7061** in either direction, and `apache`'s ROC AUC is below chance in
both: **0.4258** and **0.3425**.

**One view equals or beats every fused system reported here.** That is a
property of the label, not a result about the architecture: a task one sensor
solves cannot discriminate between ways of combining five. **No AIT number in
this report is evidence about fusion** (L17), and the same fact conditions
Part 29 — an attack result that does not say whether `dns` was in the
compromised set is uninterpretable.

### 4.3 LANL cyber1 — clean-data evidence only

Part 30 ran four view configurations on a second dataset family. **Part 29's
test battery was not run here**: no chorus attack (test B), no masked-view
detector AUC (test D), and only half of the honest-duplication test (C).
**No robustness or attack claim carries over to LANL** (L30).

| arm | system | balanced accuracy | ROC AUC | Brier | ECE |
|---|---|---|---|---|---|
| `with_auth` | naive | 0.5587 +/- 0.0088 | 0.8278 +/- 0.0033 | 0.0370 +/- 0.0005 | 0.0084 +/- 0.0009 |
| `with_auth` | PrismFlow | 0.5676 +/- 0.0073 | 0.8375 +/- 0.0043 | 0.0366 +/- 0.0002 | 0.0076 +/- 0.0011 |
| `without_auth` | naive | 0.5260 +/- 0.0006 | 0.7755 +/- 0.0025 | 0.1218 +/- 0.0006 | 0.0947 +/- 0.0011 |
| `without_auth` | PrismFlow | 0.5258 +/- 0.0016 | 0.7775 +/- 0.0030 | 0.1219 +/- 0.0005 | 0.0943 +/- 0.0014 |
| `without_dns` | naive | 0.5617 +/- 0.0093 | 0.8336 +/- 0.0029 | 0.0386 +/- 0.0007 | 0.0101 +/- 0.0007 |
| `without_dns` | PrismFlow | 0.5591 +/- 0.0051 | 0.8389 +/- 0.0058 | 0.0386 +/- 0.0002 | 0.0093 +/- 0.0004 |
| `without_flows` | naive | 0.5552 +/- 0.0049 | 0.8137 +/- 0.0022 | 0.0628 +/- 0.0004 | 0.0330 +/- 0.0008 |
| `without_flows` | PrismFlow | 0.5517 +/- 0.0067 | 0.8233 +/- 0.0031 | 0.0630 +/- 0.0006 | 0.0320 +/- 0.0007 |

ENIV on the `with_auth` arm (PrismFlow, eigen form) is **3.5627 +/- 0.0429**
against four nominal views, efficiency ratio **0.8907 +/- 0.0107**, mean
dependence **0.1363 +/- 0.0140**.

An untrained single scalar reads most of that ranking.
`total_records_across_views`, oriented on the train block only and never
fitted, reaches test ROC AUC **0.6843** against the **0.7755** to **0.8389**
the eight arm-and-system rows above span (L26).

Four facts govern how the table is read. None of them is a detection result;
all are properties of this sample and this split:

- **Split by whether the attack computer is also attacked in train, the
  trained arms' lead disappears** (L26). The scalar reaches **0.6405** on the
  **255** seen attack samples and **0.7563** on the **155** unseen ones, while
  the arms reach **0.8153** to **0.8875** on seen and **0.7102** to **0.7614**
  on unseen. No arm clears the scalar by more than **0.01** on unseen
  computers.
- **One host carries a third of the attack class** (L27). `C17693` accounts
  for **279** attack samples — **142** of the **439** in train and **137** of
  the **410** in test — with **1** benign sample in each block. Dropping its
  test rows lowers test ROC AUC by **0.036** to **0.052** depending on arm and
  system (`results/lanl/results.json`,
  `phase4_closeout.drop_top_test_computer.per_arm`, each arm 5 seeds,
  differenced against the eight rows of the table above;
  `docs/KNOWN_LIMITATIONS.md` L27).
- **Auth presence alone is a selection signal** (L24). Every attack sample has
  an `auth` record, so the untrained `auth_present_indicator` reaches test AUC
  **0.6637**. The `without_auth` arm removes the auth features; it does not
  remove that selection.
- **The `without_auth` arm is capped before training** (L24). **139** of its
  **410** test attack samples have no record in any remaining view, so they
  are unpredictable by construction.

Three further qualifications travel with every LANL number:

- **The sample is a declared case-control sample, not the network** (L23).
  Controls are a deterministic 1/512 hash of the control population; the test
  prevalence is **0.0219**, while the population prevalence implied by the
  occupied-cell bounds is far smaller. Every threshold metric above is
  computed at the sample's prevalence, not the network's.
- **The execution mode changes the fitted weights, so the per-seed std
  understates the spread** (L28). The same seed differs by up to **0.0140** in
  ROC AUC between modes — naive seed 0 reads **0.8375** in one and **0.8235**
  in the other, about four per-seed standard deviations of **0.0033**.
  Differences of order 0.01 in balanced accuracy, attack recall and Brier are
  **not established**.
- **On two arms the probabilities are worse than a constant** (L29). The
  constant-prevalence Brier baseline is **0.0428**; `without_auth` scores
  **0.1218** and **0.1219**, `without_flows` **0.0628** and **0.0630**.
  Ranking is unaffected — ROC AUC stays between **0.7755** and **0.8233** —
  but those arms' probability outputs should not be used.

**Sources:** `results/ait_clean/summary.md`, `results/ait_clean/baselines.json`, `results/lanl/results.json`, `results/lanl/population.json`.

---

## 5. What works: detection

The claim, the verdicts and the rules below are reproduced from the Part 31
claim table without strengthening. Nothing is pooled across datasets and
nothing is averaged across AIT's two directions.

**Claim.** *Detection: under attack, dependence among the implicated views
rises above its clean reference.*

| dataset | verdict |
|---|---|
| synthetic | **CONFIRMS** |
| AIT Log Data Set V2.0 | **NO EVIDENCE** |
| LANL cyber1 | **NO EVIDENCE** |

**synthetic — CONFIRMS.** The rule: *CONFIRMS only if mean dependence among
the compromised views under chorus k=2 eps=1.0 exceeds clean all-pairs
dependence by more than the two 5-seed standard deviations combined.*
Measured: compromised **0.6970 +/- 0.0380** against clean
**0.1849 +/- 0.0361**, over 5 seeds.

**AIT — NO EVIDENCE.** The two directions disagree. Both readings are
PrismFlow's mean dependence on the `dnsteal_only` attack windows against the
benign windows — that subset, not all attack windows, and that system, not
naive. `A_rm_to_santos`: **0.2906 +/- 0.0274** against
**0.3273 +/- 0.0124** — it falls. `B_santos_to_rm`: **0.3619 +/- 0.0196**
against **0.2482 +/- 0.0249** — it rises. Both were measured over 5 seeds, so more
runs of the same design will not settle it, and L17 conditions both.

**LANL — NO EVIDENCE.** Test A does not hold its sign across the four arms.
PrismFlow's attack-minus-benign mean dependence: `with_auth` **+0.0473**,
`without_auth` **+0.0602**, `without_dns` **+0.0306**, `without_flows`
**-0.1092**. The last reverses, and it reverses for naive on the same arm too
(**-0.1238**, `results/lanl/results.json`
`part29_test_a_natural_attack_windows`).

**What detection amounts to.** The estimator identifies coordinated collusion
reliably **on synthetic data, where `rho` was set**. On neither real dataset
does the reading survive its own control: on AIT the two directions disagree,
on LANL the four arms disagree. Detection is evidenced on synthetic data, and
there only.

**Sources:** `web/data/real_summary.json`.

---

## 6. What does not: defence

**Claim.** *Defence: the dependence discount lowers attack success relative to
the undiscounted baseline.*

| dataset | verdict |
|---|---|
| synthetic | **CONTRADICTS** |
| AIT Log Data Set V2.0 | **NO EVIDENCE** |
| LANL cyber1 | **needs more data** |

**synthetic — CONTRADICTS.** The rule: *CONFIRMS if PrismFlow's mean attack
success is below naive's in every non-clean condition; CONTRADICTS if it is at
or above naive's in every one of them; otherwise NO EVIDENCE.* Measured:
PrismFlow is above naive in **12** of **12** non-clean conditions, 5 seeds
each. The discount did not merely fail to help — the attack succeeded slightly
*more* often with it than without it, in every attacked cell.

**AIT — NO EVIDENCE.** All **185** cells that carry an attack-success
measurement were compared per direction, and their signs are mixed: PrismFlow below naive in **117** cells and above in **68** in
direction A, below in **98** and above in **87** in direction B.

**LANL — needs more data.** Part 29's chorus grid was not run on LANL, so no
attack-success cell exists to test either way. This is an absent experiment,
not a negative result — which is why it is *needs more data* and not
*NO EVIDENCE*.

### Why it fails — the mechanism

Four properties, each recorded as a limitation. Together they are one
explanation, not four separate excuses:

- **L3 — the discount addresses redundant evidence, not unreliable evidence.**
  Noise makes a view *less* dependent, not more, so an uncoordinated
  compromise moves the dependence signal the wrong way or not at all.
- **L4 — the discount is proportional, not capping, and weakens as evidence
  grows.** Scaling colluding evidence down by a factor still lets it dominate
  once there is enough of it. Only a cap bounds its contribution.
- **L5 — the detector inherits the dependence signal's blind spot, and inverts
  under missing evidence.** Measured at AUC **0.1247 +/- 0.0319** with half
  the views missing: not degraded to chance, inverted.
- **L8 — detection and correction fail together, because both read cross-view
  belief agreement.** They are not independent safeguards. An attacker who
  defeats the signal defeats both at once, which is what the adaptive attacker
  of section 3.6 does.

**As tested, the claim that the discount yields fusion robust to colluding
views is not supported.** The defence half of the thesis is refuted on
synthetic data, and it has never been validated on AIT or on LANL.

**Sources:** `web/data/real_summary.json`, `results/final_summary.json`.

---

## 7. V2 status — semantic multi-angle analysis

V2 asked the same question of real retrieved evidence: five angles retrieve
independently, their claims are fused, and the fused confidence is discounted
by measured semantic dependence. **The question was never reached.**

### 7.1 Retrieval coverage, after the V2-L2 and V2-L3 fixes

| angle | rows with at least one record | records | connector |
|---|---|---|---|
| tech | 44 of 48 | 440 | GitHub 407 records on 44 rows, arXiv 33 on 5 rows |
| market | 6 of 48 | 11 | NewsAPI |
| financial | 0 of 48 | 0 | yfinance |
| sentiment | 1 of 48 | 3 | StackExchange |
| regulatory | 0 of 48 | 0 | no connector exists |

Postfix, **3** of 48 rows retrieve nothing, **39** retrieve exactly one angle
and **6** retrieve two; counted by claims rather than by retrieval it is
**6**, **40** and **2**. So on **46** of 48 rows there is nothing for
dependence, ENIV, the discount or fusion to act on. Retrieval is not claims.

### 7.2 The consequence

**ECE, Brier and the conflict figures on the 48-row V2 dataset are not
measured** (V2-L6). The harness *refused* to compute them rather than compute
them on a handful of rows. That refusal is the correct behaviour and is the
honest summary of V2's state.

Two rows fused at 48 rows — rows 032 and 044, at ENIV **1.7052** and
**1.657** in the post-fix 48-row pass
(`results/v2/part24_pipeline_verification_postfix.json`, `rows[30].eniv` and
`rows[41].eniv`). A single fused row is not a result: at one row, one pass,
with a lexical adjudicator standing in for Claude, row 032 shows the code path
executes end to end and nothing more. In the pre-fix 48-row pass the same row
read ENIV **1.648** with confidence **0.6731**
(`results/v2/part24_pipeline_verification_pass1.json`, `rows[30]`); in the
later 6-row re-verification it fused to **1.7634** with confidence **0.5580**
(`results/v2/part24_reverify_n6.json`, `rows[1]`). **The three differ although
the code is identical**, because the reasoner sends no temperature and no seed and the
provider samples at its own default — which is itself a reason no single pass
can be a finding.

### 7.3 The nine V2 limitations

- **V2-L1** — one angle of five contributes; four are thin or absent.
- **V2-L2** — **resolved**. GitHub returned no records because the query was
  an eight-to-twelve term conjunction.
- **V2-L3** — **resolved**. arXiv was query-independent because the connector
  was sent raw prose.
- **V2-L4** — financial is structurally incompatible with this dataset: the
  pitches are deliberately anonymised and yfinance needs a ticker.
- **V2-L5** — the free Groq tier cannot fund a 5-seed evaluation.
- **V2-L6** — consequence: ECE and Brier cannot yet be computed.
- **V2-L7** — created by the V2-L2 fix: some GitHub records exceed the whole
  token budget and are dropped whole.
- **V2-L8** — **closed, blocked on funding**. Part 20's Claude and gpt-4o
  reasoners have never returned a live response.
- **V2-L9** — **closed, not measured**. No V2 calibration figure exists, and
  clearing the harness gate would not produce one.

V2-L2 and V2-L3 were the same fault — a connector handed a bag of words where
it expects a query language — and fixing them advanced **coverage only**.

**Sources:** `results/v2/part24_pipeline_verification_postfix.json`, `results/v2/coverage_funnel_old_vs_postfix.json`, `results/v2/part24_pipeline_verification.json`, `results/v2/part24_pipeline_verification_pass1.json`, `results/v2/part24_reverify_n6.json`, `docs/v2-known-limitations.md`.

---

## 8. Limitations and threats to validity

The record is `docs/KNOWN_LIMITATIONS.md` (L1–L32) and
`docs/v2-known-limitations.md` (V2-L1 to V2-L9). This section names the
threats that bound this report's claims rather than restating all forty-one.

**A testbed is not a network.** AIT is a testbed: its traffic is generated,
its attack is one continuously-running exfiltration tool, and one view solves
the label (L17). LANL is a real enterprise network, but what was run on it is
clean-data classification only (L30). No result in this report was measured on
real adversarial traffic in a production network.

**Label leakage on LANL `auth`.** The `auth` view is label-adjacent, and
dropping it does not drop the selection (L24). The `without_auth` arm removes
the auth features, but the selection survives: every attack sample has an
`auth` record, so auth presence alone reaches test AUC **0.6637**, and **139**
of that arm's **410** test attack samples have no record in any remaining view
at all. It is also the arm whose Brier is worst.

**On LANL an untrained scalar is most of the ranking** (L26).
`total_records_across_views` reaches test ROC AUC **0.6843**; split by whether
the attack computer is also attacked in train it reaches **0.6405** on the
**255** seen attack samples and **0.7563** on the **155** unseen ones, against
**0.8153** to **0.8875** and **0.7102** to **0.7614** for the arms, and no arm
clears it by more than **0.01** on unseen computers.

**One LANL host carries a third of the attack class** (L27). `C17693` holds
**279** of the **849** attack samples with **1** benign sample per block, and
dropping its test rows lowers test ROC AUC by **0.036** to **0.052**.

**Window choice.** Both real datasets are windowed, and the window length is a
choice that changes the label distribution, the per-view missingness and the
sample space. The LANL figures are at 300-second windows over days 1 to 16;
the AIT figures are at 300-second windows in network mode. No sensitivity
analysis over window length is reported here (L31).

**Prevalence is a sampling decision on LANL** (L23). The reported prevalence
is the sample's, not the network's.

**Execution mode is a hidden factor** (L28). A 5-seed std computed inside one
run shares one execution mode and is not independent evidence about it.

**One dataset family for the synthetic claims, and five seeds that are five
partitions of it** (L14). This bounds everything in section 3.

**The clone comparison is not like-for-like** (L22): the suspicion score is a
maximum over off-diagonal pairs, so it is not comparable across different view
counts. The live service of Part 32 states this in every response.

**The detection protocol on real data cannot speak to deployment** (L18).
Every window in the evaluated batch is attacked, so there is no clean
population inside an attacked run, and no cell anywhere in the grid varies the
contaminated fraction.

**No human-subject evaluation and no deployment evaluation were run** (L15).

**The Part 32 container has never been built** (L32). The Dockerfile is
committed and unverified, so nothing in this report establishes that the live
service reproduces inside a container.

**Sources:** `docs/KNOWN_LIMITATIONS.md`, `docs/v2-known-limitations.md`.

---

## 9. Future work

**A signal that is not cross-view agreement.** This is first because L8 is the
project's structural result: detection and correction fail together because
both read the same quantity. The untested hypothesis is that timestamps,
provenance and retrieval paths carry information a belief-agreement signal
does not: that two views reading the same upstream source would leave a shared
provenance trail whether or not their beliefs were coordinated, and that an
attacker able to suppress measured dependence would not thereby suppress that
trail. Both halves are conjecture. Nothing of this kind has been built or
measured, and nothing here is known about what an attacker can or cannot
suppress.

**A capping, per-sample discount.** The defence result would be revisited by a
discount that caps a view's contribution rather than scaling it, applied per
sample rather than per batch (L4). The present mechanism's failure mode is
specific, and so is the replacement it points to.

**Part 20 verification — blocked on API funding.** The Claude and gpt-4o
reasoners have never returned a live response; the `OpenAIReasoner` *code
path* was verified against Groq, which says nothing about OpenAI's own models
(V2-L8).

**Part 24 calibration — not measured, and coverage comes first.** The order is
coverage, then seeds, then metrics, because each is a precondition for the
next. A connector whose queries match the dataset's language, for any of the
four thin angles, comes first; funded capacity for five passes second; only
then do ECE and Brier mean anything.

**Tests B, C and D on LANL**, and a test-A reading that holds its sign across
all four arms (L30).

**A label whose signal is spread across views**, so that no single view solves
it and fusion has something to do (L17).

**Sources:** `docs/KNOWN_LIMITATIONS.md`, `docs/v2-known-limitations.md`, `docs/PROJECT_STATUS.md`.

---

## 10. Not measured

These are absences, not zeros. Nothing in this list has a committed number,
and none of them is estimated anywhere in this report.

| quantity | status |
|---|---|
| ECE on the 48-row V2 dataset | **not measured** — the harness refused (V2-L6) |
| Brier score on the 48-row V2 dataset | **not measured** — the harness refused (V2-L6) |
| conflict figures on the 48-row V2 dataset | **not measured** — the harness refused (V2-L6) |
| a five-seed V2 evaluation | **not measured** — not run; the free tier funds about two passes a day (V2-L5) |
| Claude as the fusion adjudicator (Part 23) | **not measured** — the committed run used the offline lexical adjudicator |
| Claude as a reasoner (`claude_reasoner.py`) | **not measured** — never returned a live response (V2-L8) |
| gpt-4o as a reasoner | **not measured** — Groq verified the code path, not the model (V2-L8) |
| regulatory-angle retrieval | **not measured** — no connector exists in this checkout (V2-L1) |
| financial-angle retrieval on this dataset | **not measured** — structurally impossible; the pitches are anonymised (V2-L4) |
| chorus attack success on LANL (Part 29 test B) | **not measured** — not run (L30) |
| masked-view detector AUC on LANL (Part 29 test D) | **not measured** — not run (L30) |
| honest-duplication detector AUC on LANL (Part 29 test C) | **not measured** — no within-group pair exists, and the naturally-dependent pairs were not scored (L30) |
| honest-duplication detector AUC on AIT | **not measured** — `sensor_groups()` yields no within-group pair on these five views |
| a clone condition whose clean reference has the same view count | **not measured** — the reason the clone AUC cannot be differenced (L22) |
| detector AUC on partially contaminated batches | **not measured** — no cell varies the contaminated fraction (L18) |
| the cause of the LANL Brier failure on two arms | **not measured** — three candidate causes, none tested (L29) |
| seeds run across two execution modes | **not measured** — every std shares one mode (L28) |
| the union of occupied LANL population cells | **not measured** — bounded, not counted (L23) |
| window-length sensitivity on either real dataset | **not measured** — one window length per dataset (L31) |
| attack success as a function of the ENIV surface | **not measured** — no such cell exists; the two experiments' `k` axes count different things |
| human-subject evaluation | **not measured** — not run (L15) |
| deployment evaluation | **not measured** — not run (L15) |
| the Part 32 Docker image | **not measured** — the Dockerfile has not been build-tested (L32) |

**Sources:** `docs/v2-known-limitations.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/PROJECT_STATUS.md`.

---

## How to re-check this report

```
.venv/Scripts/python.exe docs/check_report_numbers.py
```

It resolves every number in this file against the files its section cites, and
fails on any it cannot find. Where it disagrees with the text, the text is
wrong.
