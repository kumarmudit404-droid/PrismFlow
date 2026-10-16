"""Part 29 result loading, shared by the three reader scripts.

`results.json` (26.94 MB) and `results.3of4units_20261014_0022.json` (18.28 MB)
are committed as gzip (`-9`) instead of raw JSON, because the convention in
`.gitignore` tracks `results/*/*.json` and the largest results file tracked
anywhere else in this repository is 0.42 MB
(`results/comparison/detector.json`). Those three sizes are the `raw_bytes` of
`results_manifest.json` and the byte count on disk, divided by 1e6 -- the same
unit README section 10 prints, which derives all three rather than quoting
these. An earlier version of this docstring gave 25.69, 17.44 and 0.40: those
are MiB, and were labelled MB.

The raw `.json` files stay on disk, so a working copy that still has them keeps
using them, and they are re-ignored by name at `.gitignore` lines 32-33 so they
cannot be committed raw alongside the `.gz`. The `.git/info/exclude` entries an
earlier version of this docstring credited were inert: `.gitignore` outranks
`$GIT_DIR/info/exclude`, so they could not override the `!results/*/*.json`
un-ignore.

`load_results` therefore:

  1. prefers the raw `.json` when it is present on disk, else falls back to
     `<name>.gz`;
  2. hashes the bytes it actually parsed -- the DECOMPRESSED json either way --
     and checks that digest against `results_manifest.json`;
  3. raises on any mismatch, and raises if the manifest itself is absent.

A gzip stream's own sha256 is NOT a stable identifier for the artefact: `gzip
-9 <file>` embeds the mtime and original filename, and Python's `gzip` at the
same level emits different header bytes from GNU gzip's. The manifest pins the
decompressed json digest, which is identical by either route and is what every
number in the README is derived from.

The writer (`run_ait_attack.py`) is unchanged and still writes raw `.json`; the
`.gz` and the manifest are produced afterwards by:

    PYTHONPATH=. .venv/Scripts/python.exe -m experiments.ait_attack.results_io \\
        --write-manifest
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path("results/ait_attack")
MANIFEST = ROOT / "results_manifest.json"

#: The files committed as gzip rather than raw json. Only these are hash-pinned
#: in the manifest; the small sidecars are committed raw and are not listed.
GZIPPED = (
    "results.json",
    "results.3of4units_20261014_0022.json",
)


class ResultsIntegrityError(RuntimeError):
    """Raised when loaded bytes do not match the digest the manifest pins."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json_bytes(path: Path) -> bytes:
    """Return the json bytes at `path`, decompressing when it is a `.gz`."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def read_manifest() -> dict:
    if not MANIFEST.is_file():
        raise ResultsIntegrityError(
            f"{MANIFEST} is missing, so no loaded file can be verified; "
            "regenerate it with "
            "`python -m experiments.ait_attack.results_io --write-manifest`"
        )
    return json.loads(MANIFEST.read_text())


def resolve(name: str) -> Path | None:
    """Prefer the raw `.json`; fall back to `<name>.gz`. None if neither exists."""
    raw = ROOT / name
    if raw.is_file():
        return raw
    packed = ROOT / f"{name}.gz"
    if packed.is_file():
        return packed
    return None


def load_results(name: str, explicit: Path | str | None = None):
    """Load one Part 29 result file, verified against the manifest.

    `name` is the logical file name (`results.json`), never the `.gz`. Pass
    `explicit` to force a particular file on disk -- used to prove the `.gz`
    path produces the same README as the raw `.json`. Returns None when the
    file is absent, matching the previous `load()` behaviour, but a file that
    IS present and fails its digest raises instead of being used.
    """
    path = Path(explicit) if explicit is not None else resolve(name)
    if path is None:
        return None
    if explicit is not None and not path.is_file():
        raise ResultsIntegrityError(f"{path} does not exist")

    raw = read_json_bytes(path)
    entries = read_manifest()["files"]
    expected = entries.get(name, {}).get("sha256_decompressed_json")
    if expected is not None:
        actual = sha256_bytes(raw)
        if actual != expected:
            raise ResultsIntegrityError(
                f"{path} decompresses to sha256 {actual} but "
                f"{MANIFEST} pins {expected} for {name}; refusing to use it"
            )
    return json.loads(raw)


def build_manifest() -> dict:
    """Measure every gzipped pair on disk. Nothing here is typed by hand."""
    files = {}
    for name in GZIPPED:
        raw_path, gz_path = ROOT / name, ROOT / f"{name}.gz"
        if not gz_path.is_file():
            raise SystemExit(f"{gz_path} is missing; create it with gzip -9 first")
        payload = read_json_bytes(gz_path)
        digest = sha256_bytes(payload)
        if raw_path.is_file():
            on_disk = sha256_bytes(raw_path.read_bytes())
            if on_disk != digest:
                raise ResultsIntegrityError(
                    f"{raw_path} (sha256 {on_disk}) and {gz_path} (decompresses "
                    f"to {digest}) are not the same json"
                )
        files[name] = {
            "name": name,
            "gz_name": f"{name}.gz",
            "raw_bytes": len(payload),
            "gz_bytes": gz_path.stat().st_size,
            "sha256_decompressed_json": digest,
        }
    return {
        "what": ("Digests for the two Part 29 result files that are committed as "
                 "gzip (-9) rather than raw json. The sha256 is of the "
                 "DECOMPRESSED json, which is identical whether the file is read "
                 "from the raw .json or through gzip.open; a gzip stream's own "
                 "digest is not stable across gzip implementations and is "
                 "deliberately not recorded."),
        "verify": ("gzip -dc results/ait_attack/results.json.gz | sha256sum"),
        "written_by": "experiments/ait_attack/results_io.py --write-manifest",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="measure the .gz files and rewrite results_manifest.json")
    parser.add_argument("--verify", action="store_true",
                        help="check every manifested file on disk against its digest")
    args = parser.parse_args()

    if args.write_manifest:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {MANIFEST}")
        for name, entry in manifest["files"].items():
            print(f"  {name}: {entry['raw_bytes']} raw -> {entry['gz_bytes']} gz, "
                  f"sha256 {entry['sha256_decompressed_json']}")
    if args.verify:
        entries = read_manifest()["files"]
        for name in entries:
            for candidate in (ROOT / name, ROOT / f"{name}.gz"):
                if not candidate.is_file():
                    print(f"  {candidate}: absent")
                    continue
                actual = sha256_bytes(read_json_bytes(candidate))
                ok = actual == entries[name]["sha256_decompressed_json"]
                print(f"  {candidate}: {'OK' if ok else 'MISMATCH'} {actual}")
                if not ok:
                    raise ResultsIntegrityError(f"{candidate} does not match the manifest")
    if not (args.write_manifest or args.verify):
        parser.error("nothing to do: pass --write-manifest or --verify")


if __name__ == "__main__":
    main()
