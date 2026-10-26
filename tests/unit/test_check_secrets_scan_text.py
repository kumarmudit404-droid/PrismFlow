"""tools/check_secrets.py still refuses what it refused before the refactor.

Part 34b moved the per-line checks out of check() into scan_text(rel, text,
secrets) so tools/push_scan.py could call the same rules instead of holding a
copy. check_secrets is the last gate before a credential reaches a commit, so
the move needs a test that fails if any of it was lost -- the patterns, the
SELF skip, the ACKNOWLEDGED pairs, the content-hash exemption, and the rule
that a finding names the variable and never the value.

Every planted key is assembled at runtime. A literal would be a real finding
in this file, and the only ways out of that would be an ACKNOWLEDGED entry or
a weaker pattern -- both spend real safety on a fixture.
"""

from __future__ import annotations

import pytest

from tools import check_secrets as cs

# One sample per entry in PROVIDER_PATTERNS, keyed by its label. Each is built
# by concatenation so no fragment here matches the pattern it exercises.
PLANTED = {
    "openai / anthropic": "sk-" + "A" * 12,
    "groq": "gsk_" + "B" * 12,
    "google": "AIza" + "C" * 24,
    "bearer token": "Bearer " + "d" * 20,
    "newsapi-shaped": "0123456789abcdef" * 2,
    "github": "ghp_" + "e" * 20,
    "slack": "xoxb-" + "1" * 12,
    "aws": "AKIA" + "F" * 16,
}


def test_every_provider_pattern_has_a_sample():
    """A new pattern without a sample would pass this file silently."""
    assert set(PLANTED) == {label for label, _ in cs.PROVIDER_PATTERNS}


@pytest.mark.parametrize("label", sorted(PLANTED))
def test_a_planted_key_is_still_flagged(label):
    line = "TOKEN = %r" % PLANTED[label]
    findings = cs.scan_text("app/settings.py", "x = 1\n" + line + "\n", {})

    assert findings, "%s shape no longer reports" % label
    assert any(f.startswith("app/settings.py:2") for f in findings), findings
    assert any(label in f for f in findings), findings
    assert not any(PLANTED[label] in f for f in findings), \
        "a finding must not echo the value"


@pytest.mark.parametrize("rel", sorted(cs.SELF | {".env"}))
def test_paths_in_self_are_still_skipped(rel):
    """These files hold the pattern definitions, or the real values."""
    # Both checks have to be live for the skip to mean anything, so the text
    # carries a planted shape AND the .env value being searched for.
    value = "not-a-real-credential-" + "y" * 10
    text = "TOKEN = %r\nOTHER = %r\n" % (PLANTED["github"], value)
    assert cs.scan_text(rel, text, {value: "REAL_TOKEN"}) == []
    assert cs.scan_text("app/settings.py", text, {value: "REAL_TOKEN"}), \
        "the same text at an unexempt path must report both checks"


def test_an_acknowledged_pair_is_still_exempt_only_at_its_own_path():
    fake = "ghp_" + "real" + "deadbeef" + "real" + "deadbeef"
    assert ("tests/v2/test_connectors.py", fake) in cs.ACKNOWLEDGED

    line = "token = %r\n" % fake
    assert cs.scan_text("tests/v2/test_connectors.py", line, {}) == []
    assert cs.scan_text("tests/v2/test_other.py", line, {}), \
        "the same value at another path must still report"


def test_the_content_hash_exemption_still_applies():
    hexrun = PLANTED["newsapi-shaped"]
    assert cs.scan_text("fixtures/api.json", '  "etag": "%s"\n' % hexrun, {}) == []
    assert cs.scan_text("fixtures/api.json",
                        '  "url": "https://x.invalid/%s"\n' % hexrun, {}) == []
    assert cs.scan_text("fixtures/api.json", '  "key": "%s"\n' % hexrun, {}), \
        "a bare 32-hex run that is not a hash or a URL must still report"


def test_an_env_value_is_reported_by_variable_name_not_value():
    secret = "not-a-real-credential-" + "z" * 10
    findings = cs.scan_text("app/settings.py", "TOKEN = %r\n" % secret,
                            {secret: "NEWSAPI_KEY"})

    assert len(findings) == 1, findings
    assert "contains the value of NEWSAPI_KEY" in findings[0]
    assert secret not in findings[0]
