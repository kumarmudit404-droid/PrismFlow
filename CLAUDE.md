# PrismFlow - Project Memory

Thesis: Agreement between views should contribute evidence in proportion to
the effective independence of those views, not in proportion to how many
views agree.

Data flow:
`views -> encoders -> evidence -> dependence -> ENIV -> discount -> fusion -> prediction + confidence + diagnostics`

## Rules

- FROZEN MODULES: previously-passing modules are frozen. If you suspect a
  defect in an earlier module, REPORT it and stop — do not fix it as a side
  effect of a later Part.
- 5-SEED RULE: every experiment runs a minimum of 5 seeds and reports mean
  and standard deviation. Single-seed results are not evidence and must
  never be reported as a finding.
- NEVER FABRICATE: never fabricate experimental results, not even as
  placeholders.

See `docs/CONTRACT.md` for the full contract (component classification,
tensor shape conventions, reproducibility rules).
