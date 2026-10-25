---
description: PrismFlow limitations programme (Parts 34-39) - order, what each addresses, what cannot be closed
---

# Limitations programme - index (Parts 34-39)

Do NOT implement anything from this file. Print it back as a checklist, then report for each
Part whether its prerequisites are MET / NOT MET.

## What "fixing a limitation" means in this project
A limitation changes only when new committed evidence changes it. Exactly three outcomes:
- RESOLVED <date>: an engineering defect is fixed, and a test or measurement shows it.
- CLOSED <date>: the experiment named under "What would change our understanding" was run,
  and the entry's evidence is updated with the numbers WHICHEVER WAY THEY FELL. A limitation
  the experiment confirms is closed as confirmed. It is never deleted.
- OPEN: not run, or run only in part (say which part).
Never reword an entry to look resolved. Never delete an entry. Every proposed change goes to
docs/proposed_limitations_update.md. docs/KNOWN_LIMITATIONS.md is edited only after the user
approves the proposal in the review chat.

## Order (one command per Claude Code session)
| Command | Addresses | Kind | Rough runtime | Review gate |
|---|---|---|---|---|
| /part34 | Part 33b fixes, L25, L32, web harness timeout, AIT config comment | engineering | 1-1.5 h | yes |
| /part35 | L23, L26 (intervals), L27 (sensitivity), L28 (with_auth), L29 | LANL re-analysis of saved scores, no training | about 1 h | yes |
| /part36 | L2, L9 (more view counts), L10, L11, L22 | synthetic estimator and detector | 2-3 h | yes |
| /part37 | L3, L4, L12, L5 | defence research, new package prismflow/v3/ | 3 h or more | yes |
| /part38 | L24, L26, L27, L28 (all arms), L30 (tests C, D, reduced B) | LANL redesign and retrain | 3-5 h, unattended | yes |
| /part39 | L16, L18, L21, L31 | AIT breadth | 3 h or more | yes |

Prerequisites: /part38 needs /part34 (L25 trainer) and /part35 merged. /part39 needs /part34.
/part34 step 3 needs Docker Desktop running on this machine.

## Not closable by Claude Code - say so, do not attempt
- L15: human-subject and deployment evaluation need participants and a deployment.
- L9 (second half): needs a real dataset with independently known dependence; none exists here.
- L14: needs unrelated multi-view benchmarks; only if the user downloads and approves them.
- L17, L19, L20: need a real label whose signal is spread across views, or more attack episodes:
  a dataset decision for the user (more AIT scenarios would help L16/L17).
- L1: a recorded property of the design (training-time adaptation), no experiment planned.
- L6, L7, L8, L13: open research. Part 37 may touch L6/L7 only if time remains. L8 needs a
  signal source (provenance, timestamps) that the synthetic generator does not model.
- V2-L1, V2-L4, V2-L5, V2-L8, V2-L9: need API funding or data sources the user must provide.
