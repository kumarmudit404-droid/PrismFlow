# Visible copy — suggestions, not changes

Nothing in this file has been applied. Each item is a proposal to approve or
reject individually.

## Scope

**In scope:** UI labels, button text, form-control labels, error strings the
reader can actually see, and section intros / eyebrow labels.

**Out of scope, and deliberately untouched:** every figure and number, every
`source:` line, every claim about what was or was not measured, every
limitation sentence, and the phrase `not measured` wherever it appears. Those
are the project's record, not its presentation. The prose of the five chapters
is also left alone — it reads as written by a person, and shortening it would
cost precision.

Items are ordered by how much they affect the reading, strongest first.

---

## 1. A build-process label is on screen

| | |
|---|---|
| **Where** | `web/theme_preview.html:17` |
| **Current** | `Theme variants — phase (f) step 1` |
| **Suggested** | `Theme variants` |
| **Reason** | `phase (f) step 1` is internal build vocabulary. It means nothing to a reader and dates the page. The history is in `web/README.md`. This is the one place where process narration is still visible rather than in a comment. |

---

## 2. The eyebrow above "Known limitations"

| | |
|---|---|
| **Where** | `web/index.html:477` |
| **Current** | `First-class section` |
| **Suggested** | `Throughout` — or drop the line entirely |
| **Reason** | Every other eyebrow on the page names where you are (`Chapter 1` … `Chapter 5`). This one instead asserts the section's importance, which is a design decision described to the reader rather than a label. The sentence underneath ("Not a footnote. Every chapter links here…") already makes the point, and makes it better. |

---

## 3. Form-control labels are lowercase

| | |
|---|---|
| **Where** | `web/scores.html:58`, `:61`, `:64` (`domain`, `angle`, `system`) and `web/assets/js/v2_dataset.js:175`, `:180–182` (`search`, `domain`, `outcome`, `conflict expected`) |
| **Current** | `domain` · `angle` · `system` · `search` · `outcome` · `conflict expected` |
| **Suggested** | `Domain` · `Angle` · `System` · `Search` · `Outcome` · `Conflict expected` |
| **Reason** | Lowercase labels read as field names copied out of a schema. These are controls a person operates, and sentence case is what every other piece of interface text on the site uses. `conflict expected` would also read better as `Conflict expected?`, since its values are yes/no. |
| **Note** | `domain`, `angle` and `system` are also used lowercase inside the explanatory sentence at `web/scores.html:69–72`, where they refer to the controls. If the labels change, that sentence should match. |

---

## 4. Button text is lowercase

| | |
|---|---|
| **Where** | `web/scores.html:67`, `web/assets/js/v2_dataset.js:182` |
| **Current** | `reset` |
| **Suggested** | `Reset filters` |
| **Reason** | Same as above, plus `reset` alone does not say what it resets — on the Scores page it sits beside three filters, and on the dataset table beside four. |

---

## 5. Error messages shown to the reader start mid-sentence

| | |
|---|---|
| **Where** | `web/assets/js/charts.js:63`, `limits.js:180`, `scores.js:567`, `v1_failures.js:227`, `v1_scenarios.js:326`, `v2_dataset.js:348` |
| **Current** | `could not load committed data — ` + the error |
| **Suggested** | `This panel could not load its committed data — ` + the error |
| **Reason** | The current string is a log line. A reader who hits it cannot tell whether the page is broken, the file is missing, or the number does not exist — and on this site that last reading matters, because `not measured` is a real and different state. Naming the panel as the subject separates "this did not load" from "this was never measured". |

| | |
|---|---|
| **Where** | `web/assets/js/v1_scenarios.js:322` |
| **Current** | `could not render — ` + the error |
| **Suggested** | `This view could not be drawn — ` + the error |
| **Reason** | Same reason. "render" is implementation vocabulary. |

---

## 6. The footer's first sentence is a fragment

| | |
|---|---|
| **Where** | `web/index.html:501`, `web/scores.html:130` |
| **Current** | `Static site. Reads committed files only — no live API calls, no credentials.` |
| **Suggested** | `This is a static site. It reads committed files only — no live API calls and no credentials.` |
| **Reason** | Two verbless fragments in a row read like a spec sheet. The content is exactly right and should not change; only the grammar is terse. |
| **Caution** | The rest of that footer sentence ("Every figure names its source file. Where a number does not exist in a committed file, this site says not measured rather than computing one.") is a claim about the record and should be left exactly as it is. |

---

## 7. Row-stepper buttons in the detail dialog

| | |
|---|---|
| **Where** | `web/assets/js/charts.js:491–492` |
| **Current** | `‹ previous row` · `next row ›` |
| **Suggested** | `‹ Previous row` · `Next row ›` |
| **Reason** | Sentence case, consistent with items 3 and 4. Low priority — these are already clear. |

---

## 8. The scrollable-table hint

| | |
|---|---|
| **Where** | `web/assets/js/main.js:53` |
| **Current** | ` (scrollable — use arrow keys)` |
| **Suggested** | ` (scrollable — use the arrow keys)` |
| **Reason** | One word. The current phrasing reads as a terminal hint. This string is appended to an accessible name, so it is heard by screen-reader users rather than seen. |

---

## 9. Table column headers are lowercase

| | |
|---|---|
| **Where** | `web/index.html` coverage table (`angle`, `rows with ≥1 record`, `records`, `source`) and the generated tables in `assets/js/` |
| **Current** | lowercase throughout |
| **Suggested** | leave as is |
| **Reason** | Listed here only so the decision is recorded rather than overlooked. Lowercase headers are a consistent, deliberate convention across every table on the site, and `source` in particular matches the `source:` lines under each figure. Changing them would be a restyle of the data presentation, not a copy fix. |

---

## Checked and deliberately left alone

- The hero, thesis and all five chapter ledes. They are written in a voice;
  they are not stiff.
- `not measured`, everywhere. Load-bearing term.
- Every `source:` line, `docs/…` citation and `results/…` path.
- Every sentence stating a limitation, a seed count, or what was not computed
  — including `Single-pass pipeline run (GROQ, LexicalAdjudicator) — not a
  Part 24 finding`, which is doing exactly the job it should.
- The nav labels `1 · Problem` … `5 · Explore`. Compact and scannable.
- The angle chips (`Tech live`, `Regulatory no connector`, `Reddit excluded`).
  Terse by design and paired with a source line underneath.
- The search placeholder `id, pitch, outcome, notes, source…`. It lists the
  fields searched, which is more useful than a prettier sentence.
