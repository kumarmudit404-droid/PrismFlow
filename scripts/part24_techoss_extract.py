#!/usr/bin/env python3
"""Part 24 Tech-OSS row extractor: Swift Evolution + Python PEPs.

Produces rows in the 8-field schema of docs/part24-dataset-schema.md and
writes them into data/v2/part24_labeled_dataset.xlsx (the human-editable
source of truth). It never writes evaluation_queries.json.

Design rules, each one a defect found during sourcing:

* PROPOSAL MATCHING IS BY NUMBER, NEVER BY TITLE OR LAST-LINK.
  Swift's `discussions` list can hold a link belonging to a different
  proposal -- SE-0012's rationale link points at SE-0097's thread. A
  rationale link is accepted only if it contains the proposal's own
  zero-padded number with no trailing digit. PEP association is structural
  (the Resolution field lives in that PEP's own file) and is additionally
  asserted against the fetched file's `PEP:` header.

* HINDSIGHT IS FLAGGED, NEVER TRIMMED.
  Some PEP abstracts were edited after the decision (PEP 408 now opens
  "Guido has rejected this PEP"). A flagged abstract is withheld from the
  workbook and reported for a human read-through.

* DATES ARE RESOLVED OR THE ROW IS DROPPED.
  outcome_date must be an exact ISO-8601 day. It comes from the RST-wrapped
  resolution date, or from the Discourse post that recorded the decision. A
  row whose date cannot be resolved exactly is dropped, never guessed.

* WITHDRAWN IS EXCLUDED.
  An author withdrawing a proposal is not the body rejecting it. Only
  explicit Rejected and Final/Accepted/implemented states are used.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKBOOK = REPO / "data" / "v2" / "part24_labeled_dataset.xlsx"
CACHE = Path(__file__).resolve().parent / ".part24_cache"

SWIFT_API = "https://download.swift.org/swift-evolution/v1/evolution.json"
PEPS_API = "https://peps.python.org/api/peps.json"
PEP_RAW = "https://raw.githubusercontent.com/python/peps/main/peps/pep-{n:04d}.rst"

UA = {"User-Agent": "PrismFlow-Part24-extractor (kumarmudit404@gmail.com)"}

# Only unambiguous states. Withdrawn/Deferred/Superseded are excluded.
SWIFT_OUTCOME = {"implemented": "Adopted", "accepted": "Adopted", "rejected": "Rejected"}
PEP_OUTCOME = {"Final": "Adopted", "Accepted": "Adopted", "Rejected": "Rejected"}

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}

# Post-decision language. Deliberately broad: a false flag costs one read,
# a missed leak puts hindsight into the benchmark.
HINDSIGHT = [
    r"\bhas\s+(?:been\s+)?rejected\b",
    r"\b(?:was|were)\s+(?:eventually\s+|later\s+|ultimately\s+)?rejected\b",
    r"\bhas\s+(?:been\s+)?accepted\b",
    r"\b(?:was|were)\s+(?:eventually\s+|later\s+|ultimately\s+)?accepted\b",
    r"\bwas\s+not\s+accepted\b",
    r"\bthis\s+(?:PEP|proposal)\s+(?:is|was|has\s+been)\s+"
    r"(?:rejected|accepted|withdrawn|declined|deferred|approved)\b",
    r"\brejection\s+notice\b",
    r"\brejected\s+(?:by|in)\s+\S",
    r"\b(?:guido|the\s+steering\s+council|the\s+bdfl)\s+ha[sd]\b",
    r"\bdeclined\b",
    r"\bwithdrawn\b",
    r"\bsuperseded\s+by\b",
    r"\bresolution\s+of\s+this\s+pep\b",
    # a link to its own decision thread sitting inside the abstract
    r"mail\.python\.org|discuss\.python\.org|forums\.swift\.org",
]


# --------------------------------------------------------------------------
# fetch, with an on-disk cache so re-runs are cheap and repeatable
# --------------------------------------------------------------------------
def fetch(url):
    CACHE.mkdir(exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9._-]", "_", url)[-150:]
    hit = CACHE / key
    if hit.exists():
        return hit.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers=UA)
    body = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    hit.write_text(body, encoding="utf-8")
    return body


def hindsight_hits(text):
    """Return the matched phrases; empty means the text reads pre-outcome."""
    out = []
    for pat in HINDSIGHT:
        for m in re.finditer(pat, text, re.I):
            out.append(m.group(0).strip())
    return sorted(set(out))


# --------------------------------------------------------------------------
# Discourse date resolution (forums.swift.org and discuss.python.org both)
# --------------------------------------------------------------------------
def discourse_post(url):
    """(iso_day, post_html, topic_title) for the post the link points at.

    /t/<slug>/<topic_id>            -> the topic's first post
    /t/<slug>/<topic_id>/<post_no>  -> that specific post

    Taking the topic date for a link that points at post 130 would record
    the day the discussion opened, not the day it was decided.
    """
    m = re.match(r"https?://([^/]+)/t/(?:[^/]+/)?(\d+)(?:/(\d+))?", url)
    if not m:
        return None, "", ""
    host, topic, post_no = m.group(1), m.group(2), m.group(3)
    want = int(post_no) if post_no else 1
    for probe in ("https://%s/t/%s/%d.json" % (host, topic, want),
                  "https://%s/t/%s.json" % (host, topic)):
        try:
            doc = json.loads(fetch(probe))
        except Exception:
            continue
        title = doc.get("title") or ""
        for p in doc.get("post_stream", {}).get("posts", []):
            if p.get("post_number") == want and p.get("created_at"):
                return p["created_at"][:10], p.get("cooked") or "", title
        if not post_no and doc.get("created_at"):
            return doc["created_at"][:10], "", title
    return None, "", ""


def discourse_date(url):
    return discourse_post(url)[0]


# Decision words as they appear in a Swift thread TITLE, e.g.
# "[Accepted] SE-0172", "Rejected SE-0009", "[Deferred] SE-0132".
TITLE_STATE = [
    ("Adopted", r"\b(accepted|approved|implemented)\b"),
    ("Rejected", r"\brejected\b"),
]
# States that are neither an adoption nor a rejection. A row whose source
# thread is titled with one of these cannot cite the label it claims.
TITLE_AMBIGUOUS = r"\b(deferred|returned\s+for\s+revision|withdrawn|review|" \
                  r"pitch|discussion|second\s+review|revised)\b"


def corroborate(url, label, source):
    """Check the cited page actually records the label. ('ok'|reason).

    Two live mismatches motivated this. SE-0132 is state=rejected in
    evolution.json but its thread is titled "[Deferred]". SE-0342 is
    state=rejected but its thread is titled "[Accepted]". Either would have
    put a contradicted label into the benchmark, sourced to a page that
    says the opposite.
    """
    iso, body, title = discourse_post(url)
    if not title and not body:
        return "source page could not be read"

    if source == "Swift":
        # The title is the decision record for swift-evolution threads.
        got = [state for state, pat in TITLE_STATE if re.search(pat, title, re.I)]
        if re.search(TITLE_AMBIGUOUS, title, re.I) and label not in got:
            return "source thread is titled %r -- does not record %s" % (
                title.strip()[:70], label)
        if not got:
            return "source thread title records no decision: %r" % title.strip()[:70]
        if label not in got:
            return "source thread title says %s, row claims %s (%r)" % (
                "/".join(got), label, title.strip()[:70])
        if len(got) > 1:
            return "source thread title is self-contradictory: %r" % title.strip()[:70]
        return "ok"

    # PEPs: the decision lives in the linked post's body, not the title.
    text = re.sub(r"<[^>]+>", " ", body)
    want = (r"\b(reject|declin|not\s+be\s+accept|will\s+not\s+be)\b"
            if label == "Rejected" else
            r"\b(accept|approv|adopt)\b")
    deny = (r"\b(accept|approv)\w*\b" if label == "Rejected" else r"\breject\w*\b")
    if not re.search(want, text, re.I):
        return "linked post does not state %s" % label
    if re.search(deny, text, re.I) and not re.search(want, text, re.I):
        return "linked post states the opposite of %s" % label
    return "ok"


# --------------------------------------------------------------------------
# PEP resolution field: unwrap the RST hyperlink shape
# --------------------------------------------------------------------------
RST_LINK = re.compile(
    r"`\s*(?P<date>\d{1,2}-[A-Za-z]{3}-\d{4})?\s*<(?P<url>https?://[^>]+)>`_+")


def unwrap_resolution(raw):
    """(clean_url, iso_date_or_None) from a PEP Resolution field.

    Two shapes appear in peps.json:
      `21-Oct-2023 <https://discuss.python.org/t/...>`__  -> url + date
      https://mail.python.org/archives/...                -> url only

    The first shape must be unwrapped or ground_truth_source fails
    load_evaluation_dataset()'s http(s) check and its no-'<' check.
    """
    if not raw:
        return None, None
    raw = raw.strip()
    m = RST_LINK.search(raw)
    if m:
        url = m.group("url").strip()
        iso = None
        if m.group("date"):
            d, mon, y = m.group("date").split("-")
            if mon[:3].title() in MONTHS:
                iso = "%s-%02d-%02d" % (y, MONTHS[mon[:3].title()], int(d))
        return url, iso
    if raw.startswith(("http://", "https://")):
        return raw.split()[0], None
    return None, None


# --------------------------------------------------------------------------
# PEP abstract extraction
# --------------------------------------------------------------------------
def pep_abstract(rst):
    """The Abstract section only, stopping at the next same-level header.

    A sed-style range over-captured PEP 637 by ~1400 words because the
    terminator list did not cover every following section name. Here a
    section ends wherever the NEXT line is an '=' underline.
    """
    lines = rst.splitlines()
    start = None
    for i in range(len(lines) - 1):
        if lines[i].strip().lower() == "abstract" and set(lines[i + 1].strip()) == {"="}:
            start = i + 2
            break
    if start is None:
        return ""
    body = []
    for j in range(start, len(lines) - 1):
        nxt = lines[j + 1].strip()
        if lines[j].strip() and nxt and set(nxt) == {"="} and len(nxt) >= 3:
            break
        body.append(lines[j])
    return clean_rst("\n".join(body))


def clean_rst(text):
    """Strip RST/Markdown markup so the pitch is plain prose.

    The pitch is what the pipeline reasons over, so leftover markup is
    noise in the input. Roles carrying an explicit target must lose the
    target and keep the label, or PEP 833's pitch ships the literal string
    "<packaging:simple-repository-html-serialization>".
    """
    text = re.sub(r"\.\.\s+code-block::.*", " ", text)
    text = re.sub(r"\.\.\s+\w[\w-]*::.*", " ", text)
    text = re.sub(r":pep:`(\d+)`", r"PEP \1", text)
    # role with an explicit target: :ref:`label <target>` -> label
    text = re.sub(r":[a-z:]+:`\s*~?([^`<]+?)\s*<[^>`]*>`", r"\1", text)
    # plain role: :class:`~mod.Thing` -> mod.Thing (drop the ~ abbreviation)
    text = re.sub(r":[a-z:]+:`\s*~?([^`]+?)\s*`", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"`([^`<]+?)\s*<[^>]*>`_+", r"\1", text)
    text = re.sub(r"\[[\w#-]+\]_", "", text)
    # Markdown inline link (Swift summaries use these): [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\(\s*[^)]*\)", r"\1", text)
    text = re.sub(r"_{2,}|\|", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def shorten(text, limit=900):
    """Trim on a sentence boundary, never mid-word. Reports if it trimmed."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    dot = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if dot > limit * 0.4:
        return cut[:dot + 1].strip(), True
    return cut.rsplit(" ", 1)[0].strip() + " ...", True


# --------------------------------------------------------------------------
# Swift
# --------------------------------------------------------------------------
def swift_rationale(prop):
    """The proposal's OWN rationale link, matched by number.

    Returns (link, note). SE-0012 carries a rationale link belonging to
    SE-0097; the number guard rejects it rather than mislabel the row.
    """
    m = re.match(r"SE-0*(\d+)$", prop["id"].strip(), re.I)
    if not m:
        return None, "unparseable proposal id"
    num = int(m.group(1))
    guard = re.compile(r"se-0*%d(?!\d)" % num, re.I)
    rats = [d for d in (prop.get("discussions") or [])
            if (d.get("name") or "").strip().lower() == "rationale"]
    if not rats:
        return None, "no rationale link"
    matched = [d["link"] for d in rats if guard.search(d["link"])]
    if not matched:
        return None, "rationale link belongs to another proposal: " + rats[-1]["link"]
    note = ("%d rationale links, took the number-matched one" % len(rats)
            if len(rats) > 1 else "")
    return matched[0], note


def swift_candidates():
    props = json.loads(fetch(SWIFT_API))["proposals"]
    out, rejected_links = [], []
    for p in props:
        state = (p.get("status") or {}).get("state")
        if state not in SWIFT_OUTCOME:
            continue
        raw = p.get("summary") or ""
        summary = clean_rst(raw)
        if not summary:
            continue
        link, note = swift_rationale(p)
        if not link:
            rejected_links.append((p["id"].strip(), note))
            continue
        out.append({
            "source": "Swift", "ref": p["id"].strip(),
            "outcome": SWIFT_OUTCOME[state], "pitch": summary,
            "url": link, "iso": None,
            # Screen the RAW summary, not the cleaned pitch. clean_rst strips
            # the markdown link, which removes the very forums.swift.org URL
            # that marks a summary as edited after the decision -- SE-0342
            # slipped through once that way.
            "flags": hindsight_hits(raw), "note": note,
        })
    return out, rejected_links


# --------------------------------------------------------------------------
# PEPs
# --------------------------------------------------------------------------
def pep_candidates(require_inline_date=True):
    peps = json.loads(fetch(PEPS_API))
    out = []
    for v in peps.values():
        if v.get("type") != "Standards Track":
            continue
        if v.get("status") not in PEP_OUTCOME:
            continue
        url, iso = unwrap_resolution(v.get("resolution") or "")
        if not url:
            continue
        if require_inline_date and not iso:
            continue
        out.append({
            "source": "PEP", "ref": "PEP %s" % v["number"],
            "number": int(v["number"]), "outcome": PEP_OUTCOME[v["status"]],
            "url": url, "iso": iso, "raw_resolution": v.get("resolution"),
            "flags": [], "note": "",
        })
    return out


def hydrate_pep(c):
    """Fetch the PEP body, assert the number matches, extract and screen."""
    n = c["number"]
    try:
        rst = fetch(PEP_RAW.format(n=n))
    except Exception as e:
        c["note"] = "fetch failed: %s" % e
        return None
    head = re.match(r"\s*PEP:\s*(\d+)", rst)
    if not head or int(head.group(1)) != n:
        c["note"] = "PEP header/number mismatch -- refusing row"
        return None
    abstract = pep_abstract(rst)
    if not abstract:
        c["note"] = "no Abstract section found"
        return None
    pitch, trimmed = shorten(abstract)
    c["pitch"] = pitch
    c["flags"] = hindsight_hits(abstract)
    c["note"] = "abstract trimmed at a sentence boundary" if trimmed else ""
    return c


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
MIN_PITCH_CHARS = 120   # a two-clause summary is not something a pipeline can reason over


def spread_order(n):
    """Every index 0..n-1, ordered by repeated bisection. Deterministic.

    The whole traversal is spread, not just the first k. An earlier version
    took k evenly spaced indices and then fell back to ascending order, so
    when the spread picks failed a quality check every replacement came from
    the front of the bucket -- three of four Swift rows landed in the same
    fortnight of 2016. Bisection keeps replacements spread too.
    """
    if n <= 0:
        return []
    out, seen = [], set()

    def take(i):
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)

    take(0)
    take(n - 1)
    segs = [(0, n - 1)]
    while segs:
        nxt = []
        for a, b in segs:
            if b - a < 2:
                continue
            m = (a + b) // 2
            take(m)
            nxt += [(a, m), (m, b)]
        segs = nxt
    for i in range(n):
        take(i)
    return out


def pick(bucket, k, datefn, log):
    """Take k dated, corroborated, substantive rows, spread across the bucket."""
    queue = spread_order(len(bucket))
    out = []
    for i in queue:
        if len(out) == k:
            break
        c = bucket[i]
        if c["flags"]:
            log.append((c["ref"], "hindsight-flagged: %s" % ", ".join(c["flags"])))
            continue
        if len(c["pitch"]) < MIN_PITCH_CHARS:
            log.append((c["ref"], "pitch too short (%d chars)" % len(c["pitch"])))
            continue
        # A summary ending in ':' pointed at a code block that is not in the
        # pitch -- SE-0347 reads as an unfinished sentence.
        if c["pitch"].rstrip().endswith((":", ",", ";")):
            log.append((c["ref"], "pitch ends mid-thought (dangling punctuation)"))
            continue
        iso = c.get("iso") or datefn(c["url"])
        if not iso:
            log.append((c["ref"], "outcome_date could not be resolved exactly"))
            continue
        why = corroborate(c["url"], c["outcome"], c["source"])
        if why != "ok":
            log.append((c["ref"], "label not corroborated: " + why))
            continue
        c["iso"] = iso
        out.append(c)
    return out


def build_rows(n_per_cell=4):
    """Return (rows, flagged, skipped). Balanced 4x Swift/PEP x Adopted/Rejected."""
    skipped, flagged = [], []

    sw, _ = swift_candidates()
    for c in sw:
        if c["flags"]:
            flagged.append(c)
    peps = [c for c in pep_candidates(require_inline_date=False)
            if c["iso"] or "discuss.python.org" in c["url"]]
    hydrated = []
    for c in sorted(peps, key=lambda x: x["number"]):
        h = hydrate_pep(c)
        if h is None:
            skipped.append((c["ref"], c["note"]))
            continue
        if h["flags"]:
            flagged.append(h)
        hydrated.append(h)

    cells = []
    for pool, src in ((sw, "Swift"), (hydrated, "PEP")):
        for outcome in ("Adopted", "Rejected"):
            bucket = [c for c in pool if c["outcome"] == outcome]
            bucket.sort(key=lambda c: c.get("number") or int(
                re.search(r"(\d+)", c["ref"]).group(1)))
            got = pick(bucket, n_per_cell, discourse_date, skipped)
            cells.append((src, outcome, len(bucket), got))

    rows = []
    for src, outcome, avail, got in cells:
        for c in got:
            note = "Source: %s %s" % (
                "Swift Evolution" if src == "Swift" else "Python",
                c["ref"] if src == "Swift" else c["ref"].replace("PEP ", "PEP "))
            if c.get("note"):
                note += " | " + c["note"]
            rows.append({
                "idea_pitch": c["pitch"],
                "domain": "Tech-OSS",
                "actual_outcome": outcome,
                "outcome_date": c["iso"],
                "ground_truth_source": c["url"],
                "conflict_expected": "",     # annotator judgment -- never auto-filled
                "notes": note,
                "_cell": "%s/%s" % (src, outcome),
                "_avail": avail,
            })
    return rows, flagged, skipped, cells


# --------------------------------------------------------------------------
# workbook writer
# --------------------------------------------------------------------------
def write_workbook(rows, start_id="003", dry_run=True):
    """Fill consecutive empty Dataset rows starting at start_id.

    Refuses to overwrite any row whose idea_pitch is already filled, so a
    re-run cannot clobber hand-labeled work.
    """
    import openpyxl
    wb = openpyxl.load_workbook(WORKBOOK)
    ws = wb["Dataset"]
    header = [c.value for c in ws[1]]
    col = {h: i + 1 for i, h in enumerate(header)}

    target = None
    for r in range(2, ws.max_row + 1):
        if str(ws.cell(r, col["id"]).value or "").strip() == start_id:
            target = r
            break
    if target is None:
        raise SystemExit("id %s not found in the workbook" % start_id)

    written = []
    r = target
    for row in rows:
        while r <= ws.max_row and (ws.cell(r, col["idea_pitch"]).value or "").strip():
            r += 1
        if r > ws.max_row:
            raise SystemExit("ran out of empty rows at sheet row %d" % r)
        rid = str(ws.cell(r, col["id"]).value or "").strip()
        for field in ("idea_pitch", "domain", "actual_outcome", "outcome_date",
                      "ground_truth_source", "conflict_expected", "notes"):
            if not dry_run:
                ws.cell(r, col[field]).value = row[field]
        written.append((rid, r, row))
        r += 1

    if not dry_run:
        wb.save(WORKBOOK)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually save the workbook (default is a dry run)")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--start-id", default="003")
    a = ap.parse_args()

    rows, flagged, skipped, cells = build_rows(a.per_cell)

    print("=== cell fill ===")
    for src, outcome, avail, got in cells:
        print("  %-14s %-9s available=%-4d selected=%d"
              % (src, outcome, avail, len(got)))

    written = write_workbook(rows, a.start_id, dry_run=not a.write)
    print()
    print("=== rows %s ===" % ("WRITTEN" if a.write else "(dry run, nothing saved)"))
    for rid, sheet_row, row in written:
        print("  id %s (sheet r%d)  %-9s %s  %s"
              % (rid, sheet_row, row["actual_outcome"], row["outcome_date"],
                 row["notes"]))

    bal = collections.Counter(r["actual_outcome"] for r in rows)
    print()
    print("=== balance === Adopted %d / Rejected %d (n=%d)"
          % (bal["Adopted"], bal["Rejected"], len(rows)))

    print()
    print("=== hindsight-FLAGGED, withheld, need a human read-through: %d ==="
          % len(flagged))
    for c in flagged:
        print("  %-9s %-9s flags=%s" % (c["ref"], c["outcome"], ", ".join(c["flags"])))

    reasons = collections.Counter(re.sub(r"\d+", "N", why) for _, why in skipped)
    print()
    print("=== skip reasons (top) ===")
    for why, n in reasons.most_common(6):
        print("  %-52s %d" % (why[:52], n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
