"""Part 26: selectively fetch one AIT Log Data Set v2.0 scenario from Zenodo.

WHY THIS IS NOT A PLAIN DOWNLOAD
--------------------------------
The per-scenario archives on Zenodo record 5789064 run from 7.1 GB
(russellmitchell) to 26.5 GB (wilson), and the great majority of that is raw
packet capture (`log.pcap.*`) that this Part does not parse. The files Part 26
actually reads -- the labelled log sources under `gather/`, the official
`labels/` tree, and the scenario metadata -- are a small fraction of the
archive and highly compressible.

Zenodo's file endpoint honours HTTP range requests (verified: a ranged GET
returns `206 Partial Content`). A ZIP archive can therefore be read like a
random-access filesystem: the End Of Central Directory record sits at the tail,
the central directory it points at lists every member with its compressed size
and local-header offset, and each member's bytes can be fetched on their own.
So this script downloads the central directory once and then fetches only the
members that match the Part's source registry.

For russellmitchell that is tens of megabytes instead of 7.1 GB. If the server
ever stops honouring ranges, `fetch_scenario` raises rather than silently
falling back to a multi-gigabyte download -- the caller has to be told the size
first (Part 26 brief, Step 1).

INTEGRITY
---------
Every extracted member is checked against the CRC-32 recorded in the archive's
own central directory before it is written, and its SHA-256 is recorded in
`manifest.json` alongside the Zenodo record ID, the archive name and the
archive size. A CRC mismatch is an error, never a warning: a truncated or
mis-assembled range response would otherwise produce a plausible-looking log
file and a wrong dataset statistic downstream.

DOWNLOADED CONTENT IS UNTRUSTED
-------------------------------
Member names come from a remote archive, so they are validated before any path
is built from them (`safe_extract_path`): absolute paths, drive letters, UNC
prefixes and any `..` component are rejected. Nothing here unpickles, executes
or `torch.load`s downloaded content; members are decompressed to bytes and
written as files. `.fdmdownload` entries (partial Free Download Manager
artefacts) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

# Run directly from a clone (`python scripts/fetch_ait.py`), so the repo root
# has to be importable before `prismflow.data.ait_lds` can be reached.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ZENODO_RECORD = 5789064
ZENODO_API = "https://zenodo.org/api/records/{record}"
ZENODO_FILE = "https://zenodo.org/records/{record}/files/{filename}"

DEFAULT_ROOT = Path("data/raw/ait")
REQUEST_TIMEOUT_S = 600
# Zenodo advertises x-ratelimit-limit: 133 (per hour) on the file endpoint.
# One member is one request, so a pause between members keeps a full fetch
# inside the budget without needing to parse the rate-limit headers.
PAUSE_BETWEEN_MEMBERS_S = 0.5
RETRIES = 4

# Scenarios published under record 5789064. Sizes are verified against the
# Zenodo API at fetch time; they are listed here only so `--list-scenarios`
# works offline.
SCENARIOS = (
    "russellmitchell",
    "santos",
    "fox",
    "harrison",
    "wardbeck",
    "shaw",
    "wheeler",
    "wilson",
)


# ---------------------------------------------------------------------------
# ZIP central directory, read over HTTP ranges
# ---------------------------------------------------------------------------

EOCD_SIG = b"PK\x05\x06"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"
LOC_SIG = b"PK\x03\x04"
ZIP64_MARKER = 0xFFFFFFFF
# The EOCD is 22 bytes plus a comment of up to 65535; the zip64 EOCD and its
# locator sit just before it. 128 KiB from the tail covers every case.
TAIL_BYTES = 128 * 1024


@dataclass(frozen=True)
class ZipMember:
    """One entry of the archive's central directory."""

    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    header_offset: int


class RangeNotSupported(RuntimeError):
    """The file endpoint answered a ranged GET with the whole file."""


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET `url`, optionally bytes [start, end] inclusive, with retries.

    A ranged request that comes back as HTTP 200 means the server ignored the
    range and is about to stream the entire archive; that is raised as
    `RangeNotSupported` instead of being read.
    """
    last: Exception | None = None
    for attempt in range(RETRIES):
        request = urllib.request.Request(url)
        if start is not None:
            request.add_header("Range", f"bytes={start}-{'' if end is None else end}")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                if start is not None and response.status != 206:
                    raise RangeNotSupported(
                        f"{url} answered a ranged GET with HTTP {response.status}; "
                        "the whole archive would be downloaded. Aborting -- report "
                        "the archive size to the user before fetching it in full."
                    )
                return response.read()
        except RangeNotSupported:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts: {last}")


def _zip64_extra(extra: bytes, wanted: Sequence[str]) -> dict[str, int]:
    """Read the zip64 extended-information field (header id 0x0001).

    `wanted` names the fields that were 0xFFFFFFFF in the fixed-size record,
    in the order zip64 stores them: uncompressed size, compressed size,
    header offset, disk start. Only the overflowed fields are present, so the
    order matters and the caller must pass exactly those.
    """
    out: dict[str, int] = {}
    i = 0
    while i + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[i : i + 4])
        body = extra[i + 4 : i + 4 + size]
        if header_id == 0x0001:
            offset = 0
            for field in wanted:
                if offset + 8 > len(body):
                    break
                out[field] = struct.unpack("<Q", body[offset : offset + 8])[0]
                offset += 8
            break
        i += 4 + size
    return out


def parse_central_directory(blob: bytes) -> list[ZipMember]:
    """Parse a central-directory blob into `ZipMember`s, zip64 included."""
    members: list[ZipMember] = []
    i = 0
    while i + 46 <= len(blob):
        if blob[i : i + 4] != CEN_SIG:
            break
        (
            _version_made,
            _version_needed,
            _flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed,
            uncompressed,
            name_len,
            extra_len,
            comment_len,
            _disk,
            _internal,
            _external,
            header_offset,
        ) = struct.unpack("<HHHHHHIIIHHHHHII", blob[i + 4 : i + 46])

        name = blob[i + 46 : i + 46 + name_len].decode("utf-8", "surrogateescape")
        extra = blob[i + 46 + name_len : i + 46 + name_len + extra_len]

        overflowed = [
            field
            for field, value in (
                ("uncompressed_size", uncompressed),
                ("compressed_size", compressed),
                ("header_offset", header_offset),
            )
            if value == ZIP64_MARKER
        ]
        if overflowed:
            patch = _zip64_extra(extra, overflowed)
            uncompressed = patch.get("uncompressed_size", uncompressed)
            compressed = patch.get("compressed_size", compressed)
            header_offset = patch.get("header_offset", header_offset)

        members.append(
            ZipMember(
                name=name,
                method=method,
                crc32=crc32,
                compressed_size=compressed,
                uncompressed_size=uncompressed,
                header_offset=header_offset,
            )
        )
        i += 46 + name_len + extra_len + comment_len
    return members


def _locate_central_directory(tail: bytes, archive_size: int) -> tuple[int, int]:
    """Return (offset, size) of the central directory from an archive tail."""
    eocd_at = tail.rfind(EOCD_SIG)
    if eocd_at < 0:
        raise RuntimeError("no End Of Central Directory record in the archive tail")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd_at + 12 : eocd_at + 20])

    if cd_offset == ZIP64_MARKER or cd_size == ZIP64_MARKER:
        eocd64_at = tail.rfind(EOCD64_SIG)
        if eocd64_at < 0:
            raise RuntimeError("archive needs zip64 but has no zip64 EOCD record")
        cd_size, cd_offset = struct.unpack("<QQ", tail[eocd64_at + 40 : eocd64_at + 56])

    if cd_offset + cd_size > archive_size:
        raise RuntimeError(
            f"central directory at {cd_offset}+{cd_size} runs past the "
            f"{archive_size}-byte archive"
        )
    return cd_offset, cd_size


def read_remote_directory(url: str, archive_size: int) -> list[ZipMember]:
    """Download just the central directory of a remote ZIP and parse it."""
    tail_start = max(0, archive_size - TAIL_BYTES)
    tail = _get(url, tail_start, archive_size - 1)
    cd_offset, cd_size = _locate_central_directory(tail, archive_size)
    blob = _get(url, cd_offset, cd_offset + cd_size - 1)
    if len(blob) < cd_size:
        raise RuntimeError(f"central directory truncated: {len(blob)} of {cd_size} bytes")
    return parse_central_directory(blob)


def fetch_member(url: str, member: ZipMember) -> bytes:
    """Range-fetch and inflate one member, verifying its CRC-32.

    The local file header's extra field can differ in length from the central
    directory's, so the header is read from the response rather than assumed:
    the fetch covers the member's data plus a generous header allowance.
    """
    if member.method not in (0, 8):
        raise RuntimeError(f"{member.name}: unsupported compression method {member.method}")

    # 30-byte fixed local header + name + extra. 64 KiB covers any real extra
    # field (the field length is a uint16).
    allowance = 30 + len(member.name.encode()) + 65535 + 16
    start = member.header_offset
    end = start + member.compressed_size + allowance - 1
    blob = _get(url, start, end)

    if blob[:4] != LOC_SIG:
        raise RuntimeError(f"{member.name}: no local file header at offset {start}")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    data_at = 30 + name_len + extra_len
    payload = blob[data_at : data_at + member.compressed_size]
    if len(payload) != member.compressed_size:
        raise RuntimeError(
            f"{member.name}: got {len(payload)} of {member.compressed_size} compressed bytes"
        )

    raw = payload if member.method == 0 else zlib.decompress(payload, -zlib.MAX_WBITS)

    if len(raw) != member.uncompressed_size:
        raise RuntimeError(
            f"{member.name}: inflated to {len(raw)} bytes, central directory says "
            f"{member.uncompressed_size}"
        )
    found = zlib.crc32(raw) & 0xFFFFFFFF
    if found != member.crc32:
        raise RuntimeError(
            f"{member.name}: CRC-32 {found:08x} does not match the archive's "
            f"{member.crc32:08x}. The range response was corrupt or the upstream "
            "archive changed; nothing was written."
        )
    return raw


# ---------------------------------------------------------------------------
# Untrusted member names
# ---------------------------------------------------------------------------


def safe_extract_path(destination: Path, name: str) -> Path:
    """Resolve `name` under `destination`, refusing anything that escapes it.

    Member names come from a remote archive and are therefore untrusted. This
    rejects absolute POSIX paths, Windows drive-qualified and UNC paths,
    backslash separators (which `PurePosixPath` would treat as an ordinary
    character, so `..\\x` would otherwise slip through on Windows), and any
    `..` component. The resolved result is then re-checked against the
    destination so a symlinked destination cannot widen it either.
    """
    if not name or name.endswith("/"):
        raise ValueError(f"not a file entry: {name!r}")
    if "\x00" in name:
        raise ValueError(f"entry name contains a NUL byte: {name!r}")
    if "\\" in name:
        raise ValueError(f"entry name contains a backslash separator: {name!r}")
    if name.startswith("/") or name.startswith("//"):
        raise ValueError(f"absolute entry name rejected: {name!r}")
    if len(name) >= 2 and name[1] == ":":
        raise ValueError(f"drive-qualified entry name rejected: {name!r}")

    pure = PurePosixPath(name)
    if pure.is_absolute():
        raise ValueError(f"absolute entry name rejected: {name!r}")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"entry name escapes the destination: {name!r}")

    base = destination.resolve()
    target = (base / pure).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"entry name escapes the destination: {name!r}")
    return target


# ---------------------------------------------------------------------------
# What Part 26 needs out of a scenario
# ---------------------------------------------------------------------------

# Paths carrying scenario provenance: the attack schedule, the labelling rules
# the official labels were generated from, and the log/server inventory.
METADATA_PREFIXES = ("processing/config/", "rules/")
METADATA_FILES = ("dataset.yaml",)

# Raw packet capture. Present in every scenario and deliberately NOT fetched:
# it is 4.4 GB compressed for russellmitchell's labelled hosts alone, and no
# Part 26 view parses it. The loader still reserves the `network` sensor group
# for it (see `prismflow.data.ait_lds`), so adding it later needs no redesign.
PCAP_FRAGMENT = "log.pcap"

SKIP_SUFFIXES = (".fdmdownload",)


def default_selector(parsed_sources: Iterable[str]) -> Callable[[str], bool]:
    """Return a predicate selecting the members Part 26 reads.

    `parsed_sources` are log-source type names from
    `prismflow.data.ait_lds.SOURCES`. A `gather/` member is taken when the
    loader's own registry recognises it as one of those types, so this script
    and the loader can never disagree about which files matter.
    """
    from prismflow.data.ait_lds import classify_log_path

    wanted = set(parsed_sources)

    def select(name: str) -> bool:
        if name.endswith("/") or name.endswith(SKIP_SUFFIXES):
            return False
        if PCAP_FRAGMENT in name:
            return False
        if name in METADATA_FILES or name.startswith(METADATA_PREFIXES):
            return True
        # The whole labels/ tree: it is the only ground truth, and it is small.
        if name.startswith("labels/"):
            return True
        if name.startswith("gather/"):
            source = classify_log_path(name)
            return source is not None and source in wanted
        return False

    return select


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def zenodo_archive(scenario: str, record: int = ZENODO_RECORD) -> tuple[str, int]:
    """Return (url, size_bytes) of `<scenario>.zip` on the Zenodo record."""
    payload = json.loads(_get(ZENODO_API.format(record=record)).decode())
    filename = f"{scenario}.zip"
    for entry in payload.get("files", []):
        if entry["key"] == filename:
            return ZENODO_FILE.format(record=record, filename=filename), int(entry["size"])
    available = sorted(e["key"] for e in payload.get("files", []))
    raise ValueError(f"{filename} is not on record {record}; available: {available}")


def fetch_scenario(
    scenario: str,
    parsed_sources: Iterable[str],
    root: Path | str = DEFAULT_ROOT,
    record: int = ZENODO_RECORD,
    dry_run: bool = False,
) -> dict:
    """Fetch the Part 26 subset of one scenario; return the manifest written.

    Members already on disk whose SHA-256 matches the recorded manifest entry
    are not re-fetched, so a resumed or repeated run is cheap and offline.
    """
    url, archive_size = zenodo_archive(scenario, record)
    destination = Path(root) / scenario
    destination.mkdir(parents=True, exist_ok=True)

    members = read_remote_directory(url, archive_size)
    select = default_selector(parsed_sources)
    wanted = sorted((m for m in members if select(m.name)), key=lambda m: m.header_offset)

    manifest_path = destination / "manifest.json"
    previous: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            previous = {
                entry["path"]: entry
                for entry in json.loads(manifest_path.read_text()).get("entries", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            previous = {}

    total_compressed = sum(m.compressed_size for m in wanted)
    total_uncompressed = sum(m.uncompressed_size for m in wanted)
    print(
        f"{scenario}: {len(wanted)} of {len(members)} entries selected, "
        f"{total_compressed / 1e6:.1f} MB compressed -> "
        f"{total_uncompressed / 1e6:.1f} MB on disk "
        f"(archive is {archive_size / 1e9:.1f} GB)"
    )
    if dry_run:
        return {
            "zenodo_record": record,
            "archive": f"{scenario}.zip",
            "archive_size_bytes": archive_size,
            "selected_entries": len(wanted),
            "archive_entries": len(members),
            "selected_compressed_bytes": total_compressed,
            "selected_uncompressed_bytes": total_uncompressed,
            "entries": [],
            "dry_run": True,
        }

    entries: list[dict] = []
    fetched = 0
    for index, member in enumerate(wanted, start=1):
        target = safe_extract_path(destination, member.name)
        known = previous.get(member.name)
        if (
            known
            and target.exists()
            and target.stat().st_size == member.uncompressed_size
            and known.get("crc32") == f"{member.crc32:08x}"
            and known.get("sha256") == hashlib.sha256(target.read_bytes()).hexdigest()
        ):
            entries.append(known)
            continue

        raw = fetch_member(url, member)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        fetched += 1
        entries.append(
            {
                "path": member.name,
                "crc32": f"{member.crc32:08x}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": member.uncompressed_size,
                "compressed_bytes": member.compressed_size,
            }
        )
        print(
            f"  [{index}/{len(wanted)}] {member.name} "
            f"({member.uncompressed_size / 1e6:.2f} MB)"
        )
        time.sleep(PAUSE_BETWEEN_MEMBERS_S)

    manifest = {
        "zenodo_record": record,
        "zenodo_url": f"https://zenodo.org/records/{record}",
        "archive": f"{scenario}.zip",
        "archive_size_bytes": archive_size,
        "archive_entries": len(members),
        "selected_entries": len(wanted),
        "selected_compressed_bytes": total_compressed,
        "selected_uncompressed_bytes": total_uncompressed,
        "parsed_sources": sorted(parsed_sources),
        "excluded": {
            "log.pcap.*": (
                "raw packet capture; no Part 26 view parses it. The `network` "
                "sensor group is reserved for it in prismflow/data/ait_lds.py."
            )
        },
        "fetch_method": "HTTP range requests against the archive's ZIP central directory",
        "integrity": "every entry verified against the archive's own CRC-32 before writing",
        "entries": sorted(entries, key=lambda e: e["path"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{scenario}: {fetched} entries fetched, {len(entries)} verified -> {destination}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", nargs="?", default="russellmitchell", choices=SCENARIOS)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--record", type=int, default=ZENODO_RECORD)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read the central directory and report the selection size, fetch nothing",
    )
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)

    if args.list_scenarios:
        print("\n".join(SCENARIOS))
        return 0

    from prismflow.data.ait_lds import PARSED_SOURCES

    try:
        fetch_scenario(
            args.scenario,
            PARSED_SOURCES,
            root=args.root,
            record=args.record,
            dry_run=args.dry_run,
        )
    except RangeNotSupported as exc:
        print(f"STOP: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
