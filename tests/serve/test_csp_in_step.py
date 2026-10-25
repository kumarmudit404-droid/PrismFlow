"""web/_headers and web/serve.py must send the same connect-src.

Part 32 ships `connect-src 'self'`, which blocks the scoring API on purpose,
and `docs/part32-live-service.md` tells a deployer to widen it in BOTH files
by hand. This test is what fails when only one of them is edited -- the
failure mode that matters, because the local server would then permit a call
the deployed site refuses (or the reverse) and the panel would appear to work
right up until it is published.

It asserts the two values are EQUAL. It does not assert what the value is,
so widening the CSP for a real deployment does not have to edit this file.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "web"

# The directive and everything up to the next `;` or end of the policy.
CONNECT_SRC = re.compile(r"connect-src[^;'\"]*(?:'[^']*'[^;'\"]*)*")

# serve.py builds its policy from adjacent string literals. Collapsing the
# join first means adding a URL on a continuation line still reads as one
# directive, instead of failing this test for a change that is correct.
LITERAL_JOIN = re.compile(r'"\s*"')


def directives(text: str) -> list[str]:
    """Every connect-src in `text`, whitespace-normalised."""
    text = LITERAL_JOIN.sub("", text)
    return [" ".join(m.group(0).split()) for m in CONNECT_SRC.finditer(text)]


def test_both_files_declare_exactly_one_connect_src():
    """Two of them in one file would make "equal" ambiguous."""
    assert len(directives((WEB / "_headers").read_text(encoding="utf-8"))) == 1
    assert len(directives((WEB / "serve.py").read_text(encoding="utf-8"))) == 1


def test_connect_src_is_identical_in_headers_and_serve():
    headers = directives((WEB / "_headers").read_text(encoding="utf-8"))[0]
    served = directives((WEB / "serve.py").read_text(encoding="utf-8"))[0]
    assert headers == served, (
        "web/_headers sends %r but web/serve.py sends %r -- "
        "docs/part32-live-service.md requires both to be edited together."
        % (headers, served)
    )
