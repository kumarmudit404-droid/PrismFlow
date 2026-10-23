"""Input gate: what an upload has to be before anything measures it.

WHAT THIS FILE IS DEFENDING AGAINST
-----------------------------------
This is the only code in the project that reads a file a stranger chose. Two
kinds of harm follow from getting it wrong, and they need different answers.

ARBITRARY CODE. ``.npy`` and ``.npz`` are pickle carriers: numpy will execute
whatever a pickled object's ``__reduce__`` says, so a plain ``np.load`` of an
upload is remote code execution. Every load here passes
``allow_pickle=False``, including inside the zip, and a pickled array is
refused rather than unwrapped.

RESOURCE EXHAUSTION. A few hundred bytes of zip can declare gigabytes of
array. The byte cap on the upload therefore proves nothing on its own, so an
``.npz``'s members are inspected through ``zipfile`` and their DECLARED
uncompressed sizes are summed BEFORE a single member is read. Parquet gets
the same treatment through its footer metadata: rows and columns are checked
from the metadata, and the row groups are only read once those pass.

REJECT, NEVER REPAIR. Nothing here fills, clips, drops or coerces. A file
that does not satisfy a rule comes back as a named error, because a service
that silently fixed an upload would be reporting a measurement of something
the user did not send.

IN MEMORY, AND NOT LOGGED. Uploads are never written to disk: every reader
takes bytes. Nothing in this module logs the filename or any part of the
contents -- the error messages name the RULE, never the data. Users upload
telemetry to a diagnostic service, and a service that echoed it into a log
would be a second copy of their data that they did not agree to.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np

# Caps. Deliberately small: this is a diagnostic demo, not a batch service,
# and every one of these is a refusal rather than a truncation.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ROWS = 20_000
MAX_VIEWS = 32
MAX_FEATURES = 256
MAX_ZIP_MEMBERS = 16

# A correlation over fewer samples than this is noise with a decimal point.
# The dependence estimator stratifies before correlating, so the floor is per
# stratum; with no model there is one stratum, which is the easy case.
MIN_SAMPLES = 30

ALLOWED_SUFFIXES = (".npz", ".npy", ".csv", ".parquet")


class GateError(ValueError):
    """An upload that will not be scored, with the rule it broke.

    `code` is stable and machine-readable; `message` is for a person. Neither
    ever contains any part of the uploaded data or its filename.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict:
        return {"error": self.code, "detail": self.message}


@dataclass
class GatedInput:
    """What survived the gate.

    features   [B, V, D] float64, finite, at least 2 views
    evidence   [B, V, K] non-negative, or None when the upload carried none
    view_mask  [B, V] boolean, or None meaning every view present
    """

    features: np.ndarray
    evidence: np.ndarray | None
    view_mask: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.features.shape)


def _suffix(filename: str | None) -> str:
    """The extension, lowercased. The name is used for nothing else.

    It is never logged and never echoed back; it only picks a reader.
    """
    name = (filename or "").lower()
    for suffix in ALLOWED_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    raise GateError(
        "unsupported_type",
        "Upload must be one of %s." % ", ".join(ALLOWED_SUFFIXES),
    )


def _check_size(blob: bytes) -> None:
    if not blob:
        raise GateError("empty_upload", "The upload is empty.")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise GateError(
            "too_large",
            "Upload exceeds the %d byte cap." % MAX_UPLOAD_BYTES,
        )


def _load_npy(blob: bytes) -> np.ndarray:
    """One array from .npy bytes, with pickle refused.

    numpy raises ValueError for a pickled array under allow_pickle=False. We
    turn it into our own named error so the caller sees a rule, not a
    traceback -- and so the refusal is visible in the response as the
    deliberate decision it is.
    """
    try:
        array = np.load(io.BytesIO(blob), allow_pickle=False)
    except ValueError as exc:
        if "allow_pickle" in str(exc) or "Object arrays" in str(exc):
            raise GateError(
                "pickle_refused",
                "The file contains a pickled Python object. Pickle is code, "
                "not data, and is never loaded here. Re-save with "
                "allow_pickle=False.",
            ) from exc
        raise GateError("unreadable", "The file is not a readable .npy.") from exc
    except Exception as exc:
        raise GateError("unreadable", "The file is not a readable .npy.") from exc
    return array


def _load_npz(blob: bytes) -> dict[str, np.ndarray]:
    """Arrays from .npz bytes, after a declared-size check on every member.

    The order matters and is the whole point: `infolist()` reads the central
    directory only, so the sizes below are known before any member is
    decompressed. Checking after the fact would mean the bomb had already
    gone off.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise GateError("unreadable", "The file is not a readable .npz.") from exc

    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise GateError(
            "too_many_arrays",
            "The archive declares %d members; the cap is %d."
            % (len(members), MAX_ZIP_MEMBERS),
        )
    declared = sum(int(m.file_size) for m in members)
    if declared > MAX_DECOMPRESSED_BYTES:
        raise GateError(
            "decompressed_too_large",
            "The archive declares %d bytes uncompressed; the cap is %d. "
            "Nothing was decompressed." % (declared, MAX_DECOMPRESSED_BYTES),
        )
    for member in members:
        if not member.filename.endswith(".npy"):
            raise GateError(
                "unexpected_member",
                "An .npz may contain .npy members only.",
            )

    out = {}
    with np.load(io.BytesIO(blob), allow_pickle=False) as handle:
        for key in handle.files:
            try:
                out[key.removesuffix(".npy")] = handle[key]
            except ValueError as exc:
                raise GateError(
                    "pickle_refused",
                    "An array in the archive is a pickled Python object. "
                    "Pickle is code, not data, and is never loaded here.",
                ) from exc
    if not out:
        raise GateError("empty_upload", "The archive contains no arrays.")
    return out


def _load_csv(blob: bytes) -> np.ndarray:
    """A 2-D table from .csv bytes, capped by row count while reading.

    max_rows is passed to the reader rather than applied afterwards, so a
    file with a hundred million rows is refused at the cap instead of being
    materialised first.
    """
    import pandas as pd

    try:
        frame = pd.read_csv(io.BytesIO(blob), nrows=MAX_ROWS + 1)
    except Exception as exc:
        raise GateError("unreadable", "The file is not a readable .csv.") from exc
    if len(frame) > MAX_ROWS:
        raise GateError(
            "too_many_rows",
            "More than %d rows; the cap is %d." % (MAX_ROWS, MAX_ROWS),
        )
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] != frame.shape[1]:
        raise GateError(
            "non_numeric",
            "Every column must be numeric; no column is parsed, coerced or "
            "dropped here.",
        )
    return numeric.to_numpy(dtype=np.float64)


def _load_parquet(blob: bytes) -> np.ndarray:
    """A 2-D table from .parquet bytes, with the footer checked first.

    `ParquetFile` reads the footer, not the row groups, so `num_rows` and the
    schema are available before any column is decoded. A file that declares
    too much is refused there, with no data read.
    """
    import pyarrow.parquet as pq

    try:
        handle = pq.ParquetFile(io.BytesIO(blob))
    except Exception as exc:
        raise GateError("unreadable", "The file is not a readable .parquet.") from exc

    meta = handle.metadata
    if meta.num_rows > MAX_ROWS:
        raise GateError(
            "too_many_rows",
            "The file declares %d rows; the cap is %d. No data was read."
            % (meta.num_rows, MAX_ROWS),
        )
    if meta.num_columns > MAX_FEATURES:
        raise GateError(
            "too_many_columns",
            "The file declares %d columns; the cap is %d. No data was read."
            % (meta.num_columns, MAX_FEATURES),
        )

    table = handle.read()
    for field in table.schema:
        import pyarrow as pa

        if not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)):
            raise GateError(
                "non_numeric",
                "Every column must be numeric; no column is parsed, coerced "
                "or dropped here.",
            )
    return np.column_stack([
        np.asarray(table.column(i), dtype=np.float64)
        for i in range(table.num_columns)
    ])


def _as_three_dim(array: np.ndarray, n_views: int | None) -> np.ndarray:
    """[B, V, D], or a named error saying why the shape cannot be one.

    A flat table has no view axis of its own, so it is only accepted with an
    explicit n_views that divides its width exactly. Guessing a split would
    be inventing the very structure the service is meant to measure.
    """
    if array.ndim == 3:
        return array
    if array.ndim == 2:
        if n_views is None:
            raise GateError(
                "shape_ambiguous",
                "A 2-D table has no view axis. Send n_views so the columns "
                "can be split, or upload a [samples, views, features] array.",
            )
        if n_views < 2:
            raise GateError("too_few_views", "At least 2 views are required.")
        if array.shape[1] % n_views:
            raise GateError(
                "shape_ambiguous",
                "%d columns do not divide into %d views."
                % (array.shape[1], n_views),
            )
        return array.reshape(array.shape[0], n_views, array.shape[1] // n_views)
    raise GateError(
        "bad_rank",
        "Expected a 2-D table or a 3-D [samples, views, features] array, got "
        "%d dimensions." % array.ndim,
    )


def _validate(features: np.ndarray) -> np.ndarray:
    """Every rule that is about the numbers rather than the container."""
    features = np.asarray(features, dtype=np.float64)
    n_samples, n_views, n_features = features.shape

    if n_views < 2:
        raise GateError(
            "too_few_views",
            "Dependence is a relation between views: at least 2 are "
            "required, got %d." % n_views,
        )
    if n_views > MAX_VIEWS:
        raise GateError(
            "too_many_views", "At most %d views; got %d." % (MAX_VIEWS, n_views))
    if n_features < 1:
        raise GateError("no_features", "Each view needs at least one feature.")
    if n_features > MAX_FEATURES:
        raise GateError(
            "too_many_columns",
            "At most %d features per view; got %d." % (MAX_FEATURES, n_features))
    if n_samples > MAX_ROWS:
        raise GateError(
            "too_many_rows", "At most %d samples; got %d." % (MAX_ROWS, n_samples))
    if n_samples < MIN_SAMPLES:
        raise GateError(
            "too_few_samples",
            "A class-conditional correlation needs at least %d samples per "
            "stratum; got %d. Fewer is noise, and the service will not "
            "report it as a measurement." % (MIN_SAMPLES, n_samples),
        )
    if not np.isfinite(features).all():
        raise GateError(
            "not_finite",
            "The array contains NaN or infinity. Nothing is imputed or "
            "dropped here; clean the data and resend it.",
        )

    # A view with no variance has no correlation with anything: the estimator
    # would divide by a zero standard deviation and return NaN for every pair
    # it touches. Naming it is more useful than a matrix of nulls.
    spread = features.std(axis=0).max(axis=-1)
    constant = np.flatnonzero(spread == 0.0)
    if constant.size:
        raise GateError(
            "constant_view",
            "View(s) %s have zero variance across samples, so no correlation "
            "with them is defined." % ", ".join(str(int(i)) for i in constant),
        )
    return features


def gate_upload(blob: bytes, filename: str | None,
                n_views: int | None = None) -> GatedInput:
    """Bytes in, a validated [B, V, D] array out, or a GateError.

    Nothing is written to disk at any point and nothing about the contents is
    logged. The filename is read once, to choose a reader, and discarded.
    """
    _check_size(blob)
    suffix = _suffix(filename)

    evidence = None
    view_mask = None

    if suffix == ".npz":
        arrays = _load_npz(blob)
        if "features" in arrays:
            raw = arrays["features"]
        elif "views" in arrays:
            raw = arrays["views"]
        elif len(arrays) == 1:
            raw = next(iter(arrays.values()))
        else:
            raise GateError(
                "no_feature_array",
                "The archive must contain an array named `features` or "
                "`views`, or exactly one array.",
            )
        evidence = arrays.get("evidence")
        view_mask = arrays.get("view_mask")
    elif suffix == ".npy":
        raw = _load_npy(blob)
    elif suffix == ".csv":
        raw = _load_csv(blob)
    else:
        raw = _load_parquet(blob)

    if raw.dtype.kind not in "fiu":
        raise GateError(
            "non_numeric", "The array must be numeric, not %s." % raw.dtype)

    features = _validate(_as_three_dim(np.asarray(raw), n_views))

    if evidence is not None:
        evidence = np.asarray(evidence, dtype=np.float64)
        if evidence.ndim != 3 or evidence.shape[:2] != features.shape[:2]:
            raise GateError(
                "evidence_mismatch",
                "`evidence` must be [samples, views, classes] matching the "
                "features' first two axes.",
            )
        if not np.isfinite(evidence).all():
            raise GateError("not_finite", "`evidence` contains NaN or infinity.")
        if (evidence < 0).any():
            raise GateError(
                "evidence_negative",
                "Evidence is non-negative by definition; got negative values.",
            )

    if view_mask is not None:
        view_mask = np.asarray(view_mask)
        if view_mask.shape != features.shape[:2]:
            raise GateError(
                "mask_mismatch",
                "`view_mask` must be [samples, views] matching the features' "
                "first two axes.",
            )
        view_mask = view_mask.astype(bool)
        if not view_mask.any(axis=0).all():
            raise GateError(
                "empty_view",
                "At least one view is absent for every sample, so it carries "
                "no evidence to measure.",
            )

    return GatedInput(features=features, evidence=evidence, view_mask=view_mask)
