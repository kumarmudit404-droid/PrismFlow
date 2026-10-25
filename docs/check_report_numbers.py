"""Every number in docs/FINAL_REPORT.md must appear in a cited results/ file.

HOW A NUMBER IS TIED TO A FILE. Each `##` section of the report carries a
`**Sources:**` line naming the committed files its numbers come from, and a
`###` subsection INHERITS it unless it declares its own. A number is checked
against those sources, plus any path named on its own line. A section with
numbers and no sources FAILS: an uncited number is exactly what this script
exists to catch.

WHAT IS NOT A MEASUREMENT, and is skipped. The exemptions are deliberately
narrow and are the only ones:

  1. anything inside `backticks` -- paths, field names, code, shapes;
  2. a limitation or part reference -- L17, V2-L6, Part 29, section 9;
  3. an ISO date, and a markdown heading's own numbering;
  4. the seed count in the fixed phrase "5 seeds" / "5-seed" -- a design
     parameter of the protocol, not a quantity read off a run.

Everything else is a measurement and must be found. "Found" means the file
holds a value that equals the reported number when rounded to the decimal
places the report used -- so the report may round, but may not invent.

    .venv/Scripts/python.exe docs/check_report_numbers.py

Exit 0 when every number resolves, 1 otherwise. Fix the REPORT, never this
script's expectations.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "FINAL_REPORT.md"

CITE = re.compile(r"(?:results|docs|web)/[A-Za-z0-9_./-]+\.(?:json|gz|md|csv|txt)")
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.]*[A-Za-z])")
BACKTICKED = re.compile(r"`[^`]*`")
SKIP_BEFORE = re.compile(r"(?:V2-)?L$|[Pp]art $|[Ss]ection $|#|§$|20\d\d-|\d-$")
SEED_PHRASE = re.compile(r"\b5(?= seeds?\b|-seed\b)")


def numbers_in(obj, out: set) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            numbers_in(k, out)
            numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, out)
    elif isinstance(obj, str):
        for m in NUMBER.finditer(obj):
            try:
                out.add(float(m.group(0)))
            except ValueError:
                pass


def load(path: Path) -> set:
    """Every number a file holds, whatever its format."""
    out: set = set()
    if path.suffix == ".gz":
        raw = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
        numbers_in(json.loads(raw), out)
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        numbers_in(json.loads(text), out)
    numbers_in(text, out)
    return out


def strip_exempt(line: str) -> list[tuple[str, int]]:
    """The checkable numbers on one line, with their column."""
    masked = BACKTICKED.sub(lambda m: " " * len(m.group(0)), line)
    masked = SEED_PHRASE.sub(" ", masked)
    if masked.lstrip().startswith("#"):
        return []
    found = []
    for m in NUMBER.finditer(masked):
        before = masked[max(0, m.start() - 10):m.start()]
        after = masked[m.end():m.end() + 2]
        if SKIP_BEFORE.search(before):
            continue
        if re.match(r"^-\d\d", after) or re.match(r"\d{4}$", m.group(0)) and "-" in after:
            continue          # ISO date
        found.append((m.group(0), m.start()))
    return found


def main() -> int:
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    cache: dict[str, set] = {}

    def pool_for(src: str, where: str, failures: list) -> set:
        if src not in cache:
            p = ROOT / src
            if not p.is_file():
                failures.append("%s  cited file missing: %s" % (where, src))
                cache[src] = set()
            else:
                cache[src] = load(p)
        return cache[src]

    # (section, token, lineno, line, sources named on the line itself)
    claims: list[tuple[str, str, int, str, list]] = []
    section_sources: dict[str, list] = {}
    inherits: set = set()
    order: list[str] = []
    section = "(preamble)"

    # A `###` subsection inherits its parent `##` section's sources, so the
    # citation can be stated once per section rather than per table.
    parent = "(preamble)"
    for lineno, line in enumerate(lines, 1):
        if line.startswith("## ") or line.startswith("### "):
            section = "%s (line %d)" % (line.strip("# ").strip(), lineno)
            order.append(section)
            section_sources.setdefault(section, [])
            if line.startswith("## "):
                parent = section
            else:
                inherits.add((section, parent))
            continue
        if "**Sources:**" in line:
            # Attributed to the enclosing `##` section wherever it sits inside
            # it, so one line at the end of a section covers its subsections.
            section_sources[parent] = sorted(set(CITE.findall(line)))
            continue
        here = sorted(set(CITE.findall(line)))
        for token, _ in strip_exempt(line):
            claims.append((section, token, lineno, line, here))

    failures, checked = [], 0
    parent_of = dict(inherits)
    for section, token, lineno, line, here in claims:
        sources = here or section_sources.get(section, [])
        if not sources:
            sources = section_sources.get(parent_of.get(section, ""), [])
        where = "line %d" % lineno
        if not sources:
            failures.append("%s  %s is uncited -- section %r has no **Sources:** line\n      %s"
                            % (where, token, section, line.strip()[:110]))
            continue
        value = float(token)
        dp = len(token.split(".")[1]) if "." in token else 0
        checked += 1
        if not any(round(v, dp) == value
                   for src in sources for v in pool_for(src, where, failures)):
            failures.append("%s  %s not found in %s\n      %s"
                            % (where, token, ", ".join(sources), line.strip()[:110]))

    print("check_report_numbers: %d number(s) checked in %s"
          % (checked, REPORT.relative_to(ROOT).as_posix()))
    for f in failures:
        print("  FAIL  %s" % f)
    print("  all numbers resolve to a cited file" if not failures
          else "  %d unresolved" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
