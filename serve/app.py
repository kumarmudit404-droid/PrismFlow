"""FastAPI app: POST /score, GET /health.

WHAT THIS SERVICE IS. A diagnostic. It measures dependence between the views
you upload, the effective number of independent views that implies, and the
discount that follows. It does not classify anything, it is not calibrated
on your data, and the defence it demonstrates is not validated -- every
response says all three, as literal fields that no result can switch off.

WHAT IT DOES WITH YOUR UPLOAD. Reads it in memory, measures it, returns
numbers, forgets it. Nothing is written to disk, nothing is logged about the
contents or the filename, and nothing is retained between requests. The
access log that `uvicorn` writes records the method and path only, which is
why no upload detail is ever put into a message that reaches a logger.

CORS IS DENY BY DEFAULT. With `ALLOWED_ORIGIN` unset the middleware is not
installed at all, so no browser origin is permitted. See
`docs/part32-live-service.md`.
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from serve.gate import MAX_UPLOAD_BYTES, GateError, gate_upload
from serve.schema import Settings, limits
from serve.scoring import (
    CALIBRATION_NOTE,
    DEFENCE_NOTE,
    ESTIMATOR_VALIDATION_NOTE,
    score,
)

SETTINGS = Settings.from_env()

app = FastAPI(
    title="PrismFlow diagnostic scoring",
    description=(
        "Measures cross-view dependence and ENIV on arrays you upload. "
        "Not a classifier, not calibrated on your data, and the defence is "
        "not validated."
    ),
    version="32.0",
)

if SETTINGS.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )


@app.get("/health")
def health() -> dict:
    """Liveness, plus the caps so a client can refuse a file before sending.

    `cors_configured` is reported as a boolean and the origins are NOT
    echoed: an unauthenticated endpoint should not tell a caller which other
    origins are trusted.
    """
    return {
        "status": "ok",
        "service": "prismflow-diagnostic-scoring",
        "version": app.version,
        "limits": limits(),
        "cors_configured": bool(SETTINGS.allowed_origins),
        "calibration": CALIBRATION_NOTE,
        "defence": DEFENCE_NOTE,
        "estimator_validation": ESTIMATOR_VALIDATION_NOTE,
    }


@app.post("/score")
async def score_upload(
    file: UploadFile = File(...),
    n_views: int | None = Form(default=None),
) -> JSONResponse:
    """Score one upload. 400 with a named rule for anything refused.

    The body is read into memory with a hard cap and never touches disk. On
    any failure the response names the RULE, never the data -- see
    `serve.gate` for why.
    """
    blob = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(blob) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "too_large",
                     "detail": "Upload exceeds the %d byte cap."
                               % MAX_UPLOAD_BYTES},
        )
    try:
        gated = gate_upload(blob, file.filename, n_views=n_views)
    except GateError as exc:
        return JSONResponse(status_code=400, content=exc.as_dict())
    finally:
        # Drop the reference promptly; the upload is not kept past the call.
        del blob

    try:
        result = score(gated, seed=SETTINGS.seed)
    except ValueError as exc:
        # A frozen-library rejection is still a rejection, and it is reported
        # as one rather than as a 500. The text comes from prismflow and
        # describes shapes, not contents.
        return JSONResponse(
            status_code=400,
            content={"error": "not_scorable", "detail": str(exc)},
        )
    return JSONResponse(status_code=200, content=result)
