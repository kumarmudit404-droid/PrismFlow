"""Every way the gate says no, and the two that are security rather than hygiene.

The pickled .npy and the zip bomb are not edge cases on a list: they are the
reasons this gate exists. Both have a test that fails loudly if the guard is
ever relaxed, including one that proves the bomb is refused from its
DECLARED size, before anything is decompressed.
"""

from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np
import pytest

from serve.gate import (
    MAX_DECOMPRESSED_BYTES,
    MIN_SAMPLES,
    GateError,
    gate_upload,
)


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def npy_bytes(array, allow_pickle=False) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=allow_pickle)
    return buf.getvalue()


def good_features(n=200, v=4, d=6, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, v, d))


def reason(blob, filename="x.npz", **kw) -> str:
    with pytest.raises(GateError) as excinfo:
        gate_upload(blob, filename, **kw)
    return excinfo.value.code


# ---------------------------------------------------------------- accepted

def test_plain_npz_is_accepted():
    gated = gate_upload(npz_bytes(features=good_features()), "x.npz")
    assert gated.shape == (200, 4, 6)
    assert gated.evidence is None


def test_npz_with_evidence_and_mask_is_accepted():
    f = good_features()
    blob = npz_bytes(
        features=f,
        evidence=np.abs(np.random.default_rng(1).normal(size=(200, 4, 3))),
        view_mask=np.ones((200, 4), dtype=bool),
    )
    gated = gate_upload(blob, "x.npz")
    assert gated.evidence.shape == (200, 4, 3)
    assert gated.view_mask.shape == (200, 4)


def test_npy_and_csv_and_parquet_round_trip():
    f = good_features()
    assert gate_upload(npy_bytes(f), "x.npy").shape == (200, 4, 6)

    flat = f.reshape(200, 24)
    import pandas as pd
    csv = pd.DataFrame(flat).to_csv(index=False).encode()
    assert gate_upload(csv, "x.csv", n_views=4).shape == (200, 4, 6)

    import pyarrow as pa
    import pyarrow.parquet as pq
    buf = io.BytesIO()
    pq.write_table(pa.table({str(i): flat[:, i] for i in range(24)}), buf)
    assert gate_upload(buf.getvalue(), "x.parquet", n_views=4).shape == (200, 4, 6)


# ------------------------------------------------------------ security: code

def test_pickled_npy_is_refused():
    """A pickled array is code. It must never be loaded, only named."""
    blob = npy_bytes(np.array([{"not": "an array"}], dtype=object),
                     allow_pickle=True)
    assert reason(blob, "evil.npy") == "pickle_refused"


def test_pickled_member_inside_an_npz_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("features.npy",
                    npy_bytes(np.array([{"a": 1}], dtype=object),
                              allow_pickle=True))
    assert reason(buf.getvalue(), "evil.npz") in {"pickle_refused", "unreadable"}


def test_a_raw_pickle_renamed_to_npy_is_refused():
    """The extension is not evidence about the contents, and is not trusted."""
    assert reason(pickle.dumps({"payload": 1}), "evil.npy") in {
        "pickle_refused", "unreadable"}


# -------------------------------------------------------- security: resources

def test_zip_bomb_is_refused_from_its_declared_size():
    """Small on the wire, enormous when expanded -- and never expanded.

    The archive below is a few KB of zeros that declare far more than the
    cap. The guard reads the central directory, sums `file_size`, and
    refuses; if it ever started decompressing first, this test would still
    pass the assertion but the process would have allocated the array, so
    the size is kept well inside what a test machine survives.
    """
    payload = np.zeros(MAX_DECOMPRESSED_BYTES // 8 + 1024, dtype=np.float64)
    raw = npy_bytes(payload)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("features.npy", raw)
    blob = buf.getvalue()
    # The point of the guard: tiny compressed, over the cap uncompressed.
    assert len(blob) < len(raw) / 100
    assert reason(blob, "bomb.npz") in {"too_large", "decompressed_too_large"}


def test_declared_size_is_checked_before_any_member_is_read():
    """A LIE in the central directory is enough to get a refusal.

    The member is one byte; the directory claims it is over the cap. Only a
    check that reads the declared size, rather than the data, can refuse
    this -- so this test fails if the guard is ever moved to post-read.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("features.npy", b"x")
    blob = bytearray(buf.getvalue())
    with zipfile.ZipFile(io.BytesIO(bytes(blob))) as zf:
        assert zf.infolist()[0].file_size == 1

    class Liar(zipfile.ZipInfo):
        pass

    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as zf:
        info = zipfile.ZipInfo("features.npy")
        zf.writestr(info, b"x")
    # Patch the stored uncompressed size in both the local header and the
    # central directory by rewriting through zipfile's own structures.
    import struct
    data = bytearray(buf2.getvalue())
    huge = MAX_DECOMPRESSED_BYTES + 1
    for marker in (b"PK\x03\x04", b"PK\x01\x02"):
        at = data.find(marker)
        assert at >= 0
        off = 22 if marker == b"PK\x03\x04" else 24
        struct.pack_into("<I", data, at + off, huge)
    assert reason(bytes(data), "liar.npz") == "decompressed_too_large"


def test_oversized_upload_is_refused():
    assert reason(b"x" * (8 * 1024 * 1024 + 1), "big.npy") == "too_large"


def test_too_many_zip_members_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(32):
            zf.writestr("a%d.npy" % i, npy_bytes(np.zeros(4)))
    assert reason(buf.getvalue(), "many.npz") == "too_many_arrays"


def test_parquet_row_cap_is_enforced_from_metadata():
    import pyarrow as pa
    import pyarrow.parquet as pq
    buf = io.BytesIO()
    pq.write_table(pa.table({"a": np.zeros(20_001), "b": np.zeros(20_001)}), buf)
    assert reason(buf.getvalue(), "big.parquet", n_views=2) == "too_many_rows"


def test_parquet_column_cap_is_enforced_from_metadata():
    import pyarrow as pa
    import pyarrow.parquet as pq
    buf = io.BytesIO()
    pq.write_table(
        pa.table({str(i): np.zeros(50) for i in range(300)}), buf)
    assert reason(buf.getvalue(), "wide.parquet", n_views=2) == "too_many_columns"


# ------------------------------------------------------------------ validity

def test_single_view_is_refused():
    assert reason(npz_bytes(features=good_features(v=1)), "x.npz") == "too_few_views"


def test_nan_is_refused_not_imputed():
    f = good_features()
    f[3, 1, 2] = np.nan
    assert reason(npz_bytes(features=f), "x.npz") == "not_finite"


def test_inf_is_refused():
    f = good_features()
    f[0, 0, 0] = np.inf
    assert reason(npz_bytes(features=f), "x.npz") == "not_finite"


def test_constant_view_is_refused():
    f = good_features()
    f[:, 2, :] = 1.0
    assert reason(npz_bytes(features=f), "x.npz") == "constant_view"


def test_too_few_samples_is_refused():
    f = good_features(n=MIN_SAMPLES - 1)
    assert reason(npz_bytes(features=f), "x.npz") == "too_few_samples"


def test_mismatched_evidence_batch_is_refused():
    blob = npz_bytes(features=good_features(),
                     evidence=np.abs(np.zeros((199, 4, 3)) + 1.0))
    assert reason(blob, "x.npz") == "evidence_mismatch"


def test_mismatched_mask_is_refused():
    blob = npz_bytes(features=good_features(),
                     view_mask=np.ones((200, 3), dtype=bool))
    assert reason(blob, "x.npz") == "mask_mismatch"


def test_negative_evidence_is_refused():
    blob = npz_bytes(features=good_features(),
                     evidence=np.full((200, 4, 3), -1.0))
    assert reason(blob, "x.npz") == "evidence_negative"


def test_view_absent_for_every_sample_is_refused():
    mask = np.ones((200, 4), dtype=bool)
    mask[:, 1] = False
    assert reason(npz_bytes(features=good_features(), view_mask=mask),
                  "x.npz") == "empty_view"


def test_unsupported_extension_is_refused():
    assert reason(b"whatever", "x.exe") == "unsupported_type"


def test_empty_upload_is_refused():
    assert reason(b"", "x.npz") == "empty_upload"


def test_two_dimensional_upload_without_n_views_is_refused():
    assert reason(npy_bytes(np.zeros((200, 24))), "x.npy") == "shape_ambiguous"


def test_n_views_that_does_not_divide_is_refused():
    assert reason(npy_bytes(np.zeros((200, 25))), "x.npy",
                  n_views=4) == "shape_ambiguous"


def test_non_numeric_csv_is_refused_not_coerced():
    csv = b"a,b,c,d\n1,2,3,hello\n" + b"1,2,3,4\n" * 200
    assert reason(csv, "x.csv", n_views=2) == "non_numeric"


def test_rank_one_upload_is_refused():
    assert reason(npy_bytes(np.zeros(200)), "x.npy") == "bad_rank"


def test_error_messages_never_echo_the_filename():
    """A refusal names the rule. It must not reflect user input back.

    Reflected filenames are how an error page becomes an injection vector in
    whatever renders it, and they are also a copy of user data in a place
    nobody audited.
    """
    secret = "patient-records-2026-Q1-CONFIDENTIAL.exe"
    with pytest.raises(GateError) as excinfo:
        gate_upload(b"data", secret)
    assert secret not in excinfo.value.message
    assert "patient" not in excinfo.value.message.lower()
