# Part 32 — the live diagnostic scoring service

A small FastAPI service that measures cross-view dependence on an array you
upload, and a "Try it" panel on the site that talks to it. **You deploy it;
Part 32 did not.** Nothing in this repository points at a running service, and
the site ships with the panel reading *Live scoring offline*.

## What it can tell you, and what it cannot

It measures dependence between the views you send, the effective number of
independent views (ENIV) that implies, and the per-view discount that follows.
Every response carries three literal fields, before any number:

| field | value |
|---|---|
| `calibration` | `uncalibrated on your data` |
| `defence` | `not validated - see KNOWN_LIMITATIONS L3, L4, L5, L8` |
| `estimator_validation` | `ENIV is validated on synthetic data only - see KNOWN_LIMITATIONS L9` |

Three further honesty constraints are built into the code rather than left to
the reader:

- **No model, so no predicted class.** The dependence estimator normally
  stratifies by the class a trained model predicts. Your upload has no model
  and no labels, so dependence is measured in **one stratum**,
  unconditionally — a weaker quantity than the experiments report. Every
  response says so in `conditioning`, and `config.conditioning_strata` is `1`.
- **No evidence, no fusion.** The discount, the fused opinion and the
  suspicion score are functions of evidence, which features alone do not
  contain. Upload an `evidence` array inside an `.npz` to get them. Without
  one they are `null` with a stated reason — never synthesised from your
  features.
- **NaN is `null`, never `0`.** A `null` in the dependence matrix means that
  pair had no measurable overlap. Rendering it as `0.0` would say
  "measured, and independent", which is the opposite.

The service reimplements nothing. It calls `feature_dependence_matrix`,
`compute_eniv`, `soft_cluster_alpha`, `evidence_discount`, `fuse_opinions` and
`suspicion_report` from the frozen `prismflow` package.

## What it does with your upload

Reads it in memory, measures it, returns numbers, forgets it.

- **Never written to disk.** Every reader takes bytes; there is no temp file.
- **Never logged.** Neither the contents nor the filename reaches a log or an
  error message. Error messages name the *rule* that was broken, never the
  data — `tests/serve/test_gate.py::test_error_messages_never_echo_the_filename`
  gates that.
- **Never retained** between requests.

## The input gate

`serve/gate.py` is the only code in this project that reads a file a stranger
chose. It is built around two harms.

**Arbitrary code.** `.npy` and `.npz` are pickle carriers: numpy will execute
whatever a pickled object's `__reduce__` says, so a plain `np.load` of an
upload is remote code execution. Every load passes `allow_pickle=False`,
including for members inside an `.npz`, and a pickled array is refused rather
than unwrapped.

**Resource exhaustion.** A few hundred bytes of zip can declare gigabytes of
array, so the byte cap proves nothing on its own. An `.npz`'s members are
inspected through `zipfile.infolist()` — the central directory, not the data —
and their **declared** uncompressed sizes are summed *before* anything is
decompressed. Parquet gets the same treatment through its footer: row and
column counts are checked from `ParquetFile.metadata` before any row group is
read.

**Reject, never repair.** Nothing is filled, clipped, dropped or coerced.

| cap | value |
|---|---|
| upload bytes | 8 MiB |
| declared uncompressed bytes (`.npz`) | 64 MiB |
| rows / samples | 20,000 |
| views | 32 |
| features per view | 256 |
| minimum samples | 30 |
| zip members | 16 |

Rejection codes: `empty_upload`, `too_large`, `unsupported_type`,
`unreadable`, `pickle_refused`, `decompressed_too_large`, `too_many_arrays`,
`unexpected_member`, `no_feature_array`, `too_many_rows`, `too_many_columns`,
`non_numeric`, `bad_rank`, `shape_ambiguous`, `too_few_views`,
`too_many_views`, `no_features`, `too_few_samples`, `not_finite`,
`constant_view`, `evidence_mismatch`, `evidence_negative`, `mask_mismatch`,
`empty_view`, `not_scorable`. Each has a test.

## Environment variables

| variable | default | what it does |
|---|---|---|
| `ALLOWED_ORIGIN` | *(unset)* | **The only origin CORS permits.** Comma-separated for more than one. **Unset means deny every browser origin** — an unconfigured deployment answers no browser at all. A literal `*` is **dropped, not honoured**, so a deployment cannot open itself to the whole internet with one character. `serve/schema.py` has no wildcard branch to fall into. |
| `SCORING_SEED` | `0` | The seed reported in every response and hashed into `config_hash`. A non-numeric value falls back to `0` rather than crashing the service. |

Set `ALLOWED_ORIGIN` to your site's **exact origin**, scheme included, with no
trailing slash — for example `https://yourname.github.io`.

**No secret, token or API key is needed by this service, and none exists
anywhere in this repository.** If a deployment ever needs one, it belongs in
the host's secret store, never in a file here.

## Deploying to Hugging Face Spaces

**The Dockerfile has not been build-tested.** No `docker build` has ever been
run against it: the Docker daemon is not running on the development machine,
so the file below is read-but-unexecuted. The first build is the deployer's,
and it may fail.

1. Create a Space, SDK **Docker**, hardware **CPU basic**.
2. Push `Dockerfile`, `serve/` and `prismflow/`. The image installs a
   **CPU-only torch wheel** from PyTorch's own index — the default PyPI wheel
   carries ~2 GB of CUDA runtime for a GPU a Space does not have — and runs as
   a non-root user (`app`, uid 1000) on port 7860.
3. In the Space's **Settings → Variables**, add `ALLOWED_ORIGIN` with your
   site's origin. Until you do, the API is reachable by `curl` but by no
   browser.
4. Check `https://<your-space>.hf.space/health` returns `{"status":"ok", …}`.

Versions in `serve/requirements.txt` are pinned exactly, not floated: a
diagnostic service whose dependency set drifts from the version that was
tested is reporting numbers nobody measured.

## Pointing the site at it — the two edits you make, by hand, after deploying

Part 32 **did not** make these. The site's Content-Security-Policy ships as
`connect-src 'self'`, which blocks the API on purpose, and the panel renders
*Live scoring offline* until you change it.

**Edit 1 — `web/_headers`.** Replace the `connect-src 'self'` term with
exactly:

```
connect-src 'self' https://<your-space>.hf.space;
```

**Edit 2 — `web/serve.py`.** Make the identical change to the CSP string it
sends, so the local server and the deployed site agree:

```
connect-src 'self' https://<your-space>.hf.space;
```

**No wildcard is acceptable in either file.** Not `connect-src *`, not
`https://*.hf.space`, not `connect-src 'self' *`. The whole value of a CSP
here is that it names the one host this page may send your uploaded data to;
a wildcard turns that guarantee into a comment. Write the exact origin, and
change it when the Space's URL changes.

**Edit 3 — tell the page where the API is.** Set the one config value before
`try_it.js` loads, in `web/index.html`:

```html
<script>window.PRISMFLOW_API_BASE = "https://<your-space>.hf.space";</script>
```

With it unset — the shipped state — the panel says *Live scoring offline* and
the rest of the site is unaffected.

### How the panel behaves when the API is not there

`try_it.js` treats **all four** of these identically, because they mean the
same thing to a reader:

- no `PRISMFLOW_API_BASE` configured (the shipped state),
- `/health` returns a non-2xx,
- `/health` times out after 4 s (a sleeping Space),
- **the browser's CSP blocks the request** — which surfaces as a `TypeError`
  from `fetch` with no response, indistinguishable from a dead host.

All four render `Live scoring offline` with a short reason, disable the
controls, and show no numbers. There is no cached result, no demo payload and
no placeholder: offline means nothing is displayed. Nothing else on the page
depends on `try_it.js`, so `web/verify_page.py` passes with the API absent —
which is how it was run for Part 32, since no service was deployed.

## The sample file

`web/data/sample_ait_views.npz` — 400 windows of per-window aggregate counts
for five sensors, **derived from the AIT Log Data Set V2.0** and therefore
carrying **CC BY-NC-SA 4.0** under that licence's share-alike term.

**It contains no raw logs**: no message text, hostname, IP address, user name,
URL or timestamp. `window_start` and the labels were deliberately left out so
the sample cannot be realigned against the source capture.

The required attribution, the two papers the Zenodo record asks to be cited,
the provenance and the file's sha256 are in
`web/data/sample_ait_views.LICENCE.md`, quoted from
`docs/part27-ait-datacard.md` section 2. Nothing there was written for this
file.

## Running it locally

```
.venv\Scripts\python.exe -m pip install -r serve/requirements.txt
set ALLOWED_ORIGIN=http://127.0.0.1:825
set PYTHONPATH=.
.venv\Scripts\python.exe -m uvicorn serve.app:app --port 7860
```

Then `python web/serve.py` for the site. The panel will still report offline
until you make edits 1–3 above, because the local server sends the same CSP
as the deployed site.

Tests: `.venv\Scripts\python.exe -m pytest -q tests/serve`.

## The container build (L32) — attempted 2026-10-26, not measured

Part 34 step 3 tried to build and run this image. **It did not run: the Docker
daemon was not reachable on this machine.** The Docker CLI is installed and
answers (client 29.7.2, context `desktop-linux`), but there is no server
behind it — `npipe:////./pipe/dockerDesktopLinuxEngine` does not exist, and
`docker ps`, `docker info --format '{{.ServerVersion}}'` and
`docker desktop status` all fail. The probes and their exit codes are in
`results/serve_docker/report.json`.

Nothing about the image is therefore known. Build success, build time, image
size, the `/health` response and the `/score` response are all **not
measured** — not zero, not assumed to work, and not estimated. **L32 stays
OPEN**, and the first `docker build` is still the deployer's.

To close it: `docker desktop start`, wait for the daemon to answer
`docker ps`, then re-run Part 34 step 3.
