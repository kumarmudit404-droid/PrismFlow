---
description: PrismFlow Part 32 - live diagnostic scoring API (serve/) plus "Try it" upload on the same web/ site
---

PART 32 - LIVE UPLOAD AND SCORING
=================================

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md, docs/v2-known-limitations.md,
web/README.md, the Part 31 section in web/index.html. Use the prismflow-honest-reporting skill.

STANDING RULES: prismflow/ FROZEN - the API imports existing functions and reimplements none;
no secrets or API keys anywhere in code, config, web files or commits; branch part32-live;
commit per step; do not push and do not deploy.

YOU MAY CREATE / MODIFY ONLY
  serve/__init__.py, serve/app.py, serve/gate.py, serve/schema.py, serve/scoring.py
  serve/requirements.txt, Dockerfile (Hugging Face Spaces, CPU)
  tests/serve/*
  web/assets/js/try_it.js (new), web/index.html (add one "Try it" section only)
  web/data/sample_ait_views.npz (small, a few hundred windows from Part 26 output) + attribution
  docs/part32-live-service.md

API (FastAPI)
1. POST /score (multipart), GET /health.
2. gate.py accepts ONLY .npz (np.load allow_pickle=False), .npy, .csv, .parquet. Size cap and row
   cap from config. Reject: V < 2, NaN/inf, mismatched B across views, too few samples for
   pairwise_holdout, constant views. Clear JSON error for each.
3. scoring.py calls feature_dependence_matrix, compute_eniv, soft_cluster_alpha,
   evidence_discount, fuse_opinions and the suspicion score - nothing reimplemented.
4. Response: dependence matrix (NaN -> null, never 0), ENIV (eigen and design-effect),
   per-view alpha, suspicion score, seed, config hash, and these literal fields:
   "calibration": "uncalibrated on your data"
   "defence": "not validated - see KNOWN_LIMITATIONS L3, L4, L5, L8"
5. CORS limited to the site's origin (from config).

WEB
6. "Try it" section: file upload, "use sample data" button, results drawn with the SAME heatmap and
   ENIV components as Part 31. Both literal caveat fields are shown, not hidden.
7. API base URL from one config value. If /health fails, show "Live scoring offline" and keep the
   rest of the site working.

TESTS
- clone views reproduce the documented eigen n_eff = 4.0 for k = 0..4
- independent views give n_eff close to V
- every gate rejection path, including a pickled .npy (must be refused)
- same seed -> identical JSON
- web: verify_page.py passes with the API offline

DONE WHEN pytest -q green; docs/part32-live-service.md explains how the user deploys the API to
Hugging Face Spaces and points the site at it (the user deploys, not you).
