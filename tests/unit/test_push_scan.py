"""tools/push_scan.py: the push set, not the working tree.

The planted key is assembled at runtime. A literal one in this file would be
caught by `check_secrets.py --all` over the tracked tree -- correctly -- and
the only ways out would be an ACKNOWLEDGED entry or a weaker pattern, both of
which spend real safety on a test fixture.
"""

from __future__ import annotations

import subprocess

import pytest

from tools import push_scan as ps

FAKE_KEY = "ghp_" + "deadbeef" * 3          # 24 chars after the prefix
LEAK_PATH = "secrets/config.py"             # not in check_secrets.SELF


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A repo whose history holds a key the working tree no longer does."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.invalid")
    git(tmp_path, "config", "user.name", "t")

    (tmp_path / "README.md").write_text("clean\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()

    leak = tmp_path / "secrets"
    leak.mkdir()
    (leak / "config.py").write_text("TOKEN = %r\n" % FAKE_KEY, encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "oops")

    (leak / "config.py").write_text("TOKEN = os.environ['TOKEN']\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "removed it")
    return tmp_path, base


def test_catches_a_key_that_the_working_tree_no_longer_has(repo):
    root, base = repo
    assert FAKE_KEY not in (root / LEAK_PATH).read_text(encoding="utf-8")

    scanned, findings = ps.scan_range(root, "%s..HEAD" % base, secrets={})

    assert len(findings) == 1, findings
    assert findings[0].startswith(LEAK_PATH + ":1")
    assert "github" in findings[0]
    assert FAKE_KEY not in findings[0], "a finding must not echo the value"
    assert scanned >= 2   # both versions of the file are in the push set


def test_a_range_with_no_leak_is_clean(repo):
    root, _ = repo
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert ps.scan_range(root, "%s~1..%s" % (head, head), secrets={}) == (1, [])


def test_trees_are_not_counted_as_blobs(repo):
    """`secrets` has no dot, so the tree passes the suffix filter.

    It must still not be read as a file: counting trees inflated an earlier
    hand-rolled version of this scan by 288 objects over this project.
    """
    root, base = repo
    pairs = ps.named_objects(root, "%s..HEAD" % base)
    assert ("secrets" in [p for _, p in pairs]
            or any(p.startswith("secrets") for _, p in pairs))
    blobs = [p for p, _ in ps.read_blobs(root, pairs)]
    assert "secrets" not in blobs
    assert all(p.endswith(".py") or p.endswith(".md") for p in blobs), blobs


def test_an_env_value_is_reported_by_variable_name_not_value(repo):
    root, base = repo
    scanned, findings = ps.scan_range(
        root, "%s..HEAD" % base, secrets={"os.environ": "FAKE_TOKEN"})
    assert any("contains the value of FAKE_TOKEN" in f for f in findings)
    assert not any("os.environ" in f.split("  ", 1)[1] for f in findings)
