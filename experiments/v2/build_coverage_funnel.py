r"""Per-angle retrieval funnel for the pre-fix and post-fix 48-row runs.

  DIAGNOSTIC, ONE PASS EACH, NOT A PART 24 FINDING

Emits one JSON holding the same five-stage funnel for both runs, so a chart can
render them side by side without recomputing anything in the browser:

    connector configured -> retrieved >=1 record -> reasoner ran
                         -> reasoner errored -> produced >=1 claim

Stage definitions, each read from a field the runs actually wrote:

  configured      the angle's warnings do NOT contain "no connector is
                  configured". Regulatory is the only angle this excludes, on
                  every row of both runs.
  retrieved       ``records > 0``.
  reasoner_ran    the angle's ``note`` is not "no evidence retrieved; reasoner
                  not called". An angle with no evidence never reaches a
                  reasoner, so this is a real stage and not a restatement of
                  the one above.
  reasoner_error  ``reasoner_error`` is set. This is a PROVIDER failure and is
                  kept separate from retrieval throughout: the two failure
                  modes have different fixes and must not be summed.
  claims          ``claims > 0``.

Also carries per-connector record totals, because "the tech angle retrieved"
and "GitHub retrieved" are different claims and the V2-L2 fix is about the
second one.

TWO RECORD COUNTS, AND WHY THEY DIFFER
--------------------------------------
``records_retrieved`` and ``records_in_context`` are NOT the same number and a
chart must never add one to the other or label them both "records":

  records_retrieved   sum of ``provenance``, which prismflow/v2/angles/base.py
                      increments after cross-source dedup but BEFORE the token
                      budget (``provenance[name] += accepted``, line 271).
  records_in_context  the harness's ``records`` field, which is
                      ``len(evidence.ranked_records)`` -- what survived
                      ``_apply_budget`` and actually reached the reasoner.

Pre-fix the two were equal (480 = 480): arXiv snippets are short and the budget
never bound. Post-fix they are not, and the gap is itself a result, so
``records_dropped_by_budget`` and ``rows_with_oversize_records`` are reported
rather than left as an unexplained discrepancy between two totals.

    .venv\Scripts\python.exe experiments/v2/build_coverage_funnel.py
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANGLES = ("tech", "market", "financial", "regulatory", "sentiment")

RUNS = (
    ("prefix", "results/v2/part24_pipeline_verification.json",
     "Pre-fix: committed 48-row run, before the Part 19 / V2-L2 GitHub fix"),
    ("postfix", "results/v2/part24_pipeline_verification_postfix.json",
     "Post-fix: 48-row re-run on current connector code, 2026-10-01"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def funnel(run: dict) -> dict:
    rows = run["rows"]
    out = {}
    for angle in ANGLES:
        configured = retrieved = ran = errored = claimed = 0
        records = 0
        oversize_rows = 0
        connectors: dict = {}
        for row in rows:
            cell = row["angles"].get(angle, {})
            warnings = cell.get("warnings") or []
            if not any("no connector is configured" in w for w in warnings):
                configured += 1
            n = cell.get("records")
            n = n if isinstance(n, int) else 0
            records += n
            if n > 0:
                retrieved += 1
            if "reasoner not called" not in (cell.get("note") or ""):
                ran += 1
            if cell.get("reasoner_error"):
                errored += 1
            c = cell.get("claims")
            if isinstance(c, int) and c > 0:
                claimed += 1
            if any("can never be included" in w
                   for w in warnings):
                oversize_rows += 1
            for source, count in (cell.get("provenance") or {}).items():
                slot = connectors.setdefault(
                    source, {"records_retrieved": 0, "rows_with_records": 0})
                slot["records_retrieved"] += count
                if count > 0:
                    slot["rows_with_records"] += 1
        retrieved_total = sum(c["records_retrieved"] for c in connectors.values())
        out[angle] = {
            "configured": configured,
            "retrieved": retrieved,
            "reasoner_ran": ran,
            "reasoner_errored": errored,
            "claims": claimed,
            # post-dedup, pre-budget; this is what `connectors` sums to
            "records_retrieved": retrieved_total,
            # post-budget; what the reasoner actually saw
            "records_in_context": records,
            "records_dropped_by_budget": retrieved_total - records,
            "rows_with_oversize_records": oversize_rows,
            "connectors": connectors,
        }
    return out


def main() -> int:
    payload = {
        "title": "Per-angle retrieval funnel, pre-fix vs post-fix",
        "is_part24_evidence": False,
        "standing": ("DIAGNOSTIC. One pass per run, seeds: null, Groq "
                     "openai/gpt-oss-120b through the OpenAIReasoner code path, "
                     "offline LexicalAdjudicator. NOT A PART 24 FINDING under "
                     "docs/CONTRACT.md section 5."),
        "stage_order": ["configured", "retrieved", "reasoner_ran",
                        "reasoner_errored", "claims"],
        "record_count_note": (
            "records_retrieved is post-dedup/pre-budget (sum of provenance); "
            "records_in_context is post-budget (len(ranked_records)), i.e. what "
            "the reasoner saw. They are different quantities -- do not add or "
            "conflate them."),
        "stage_labels": {
            "configured": "connector configured",
            "retrieved": "retrieved >=1 record",
            "reasoner_ran": "reasoner ran",
            "reasoner_errored": "reasoner errored (provider failure)",
            "claims": "produced >=1 claim",
        },
        "angle_order": list(ANGLES),
        "denominator": None,
        "runs": {},
    }
    for key, rel, caption in RUNS:
        path = ROOT / rel
        if not path.exists():
            print("missing: %s -- not written, and not estimated" % rel)
            return 1
        run = json.loads(path.read_text(encoding="utf-8"))
        payload["runs"][key] = {
            "caption": caption,
            "source": rel,
            "sha256": sha256(path),
            "n_rows": run.get("n_rows"),
            "passes": run.get("passes"),
            "seeds": run.get("seeds"),
            "is_part24_evidence": run.get("is_part24_evidence"),
            # Two different things, deliberately both reported. The first is
            # the precondition (fusion needs >=2 angles with claims); the
            # second is what actually happened. The harness writes
            # ``fused: None`` only on the early-return path and writes
            # ``eniv`` when fusion ran, so ``eniv`` presence -- not a truthy
            # ``fused`` -- is the test for a row that fused.
            "rows_reaching_two_claim_angles": [
                r["id"] for r in run["rows"]
                if (r.get("angles_with_claims") or 0) >= 2
            ],
            "rows_fused": [
                {"id": r["id"], "eniv": r.get("eniv"),
                 "n_angles": r.get("n_angles"), "discount": r.get("discount"),
                 "confidence": r.get("confidence")}
                for r in run["rows"] if "eniv" in r
            ],
            "angles": funnel(run),
        }
        payload["denominator"] = run.get("n_rows")

    out = ROOT / "results/v2/coverage_funnel_old_vs_postfix.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("written to %s" % out)
    for key in payload["runs"]:
        fused = payload["runs"][key]["rows_fused"]
        print("  %-8s fused %d/%s%s" % (
            key, len(fused), payload["denominator"],
            "".join("  [%s eniv=%s conf=%s]" % (f["id"], f["eniv"], f["confidence"])
                    for f in fused)))
        a = payload["runs"][key]["angles"]
        print("  %-8s " % key + "  ".join(
            "%s=%d/%d" % (n, a[n]["retrieved"], payload["denominator"])
            for n in ANGLES))
        print("           budget loss: " + "  ".join(
            "%s=%d of %d (%d rows oversize)"
            % (n, a[n]["records_dropped_by_budget"],
               a[n]["records_retrieved"], a[n]["rows_with_oversize_records"])
            for n in ANGLES if a[n]["records_retrieved"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
