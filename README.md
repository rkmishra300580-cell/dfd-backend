# Deepfake Detection API — Backend

FastAPI-based deepfake detection backend. Supports image, video, audio, and document analysis.

## Project structure

```
dfd-backend/
├── main.py                  # FastAPI app — endpoints /health /analyze /jobs/{id} /graph/{id}/{file} /report/{id}
├── auth.py                  # Signup/login (JWT) + API keys, via /auth/*
├── oauth.py                 # Google/Facebook OAuth login, via /auth/{provider}/*
├── billing.py                # Stripe checkout + webhook, via /billing/*
├── jobs.py                  # Async job tracking (Postgres-backed), via /jobs/*
├── rate_limit.py             # Per-IP burst limit + per-user monthly quota
├── uploads_guard.py          # Upload size + real MIME-sniffing validation
├── db.py                    # Postgres connection pool (asyncpg, Supabase pooler)
├── requirements.txt
├── Procfile                  # Render start command
└── detector/
    ├── __init__.py
    ├── config.py             # Folder paths and constants
    ├── result.py             # AnalysisResult class (builds JSON + PDF in parallel)
    ├── helpers.py            # Shared utilities: face detection, label scoring, graph style
    ├── pipeline.py           # Main dispatcher — routes file to correct analysis function
    ├── image_pipeline.py     # Image analysis (frequency + face forensics + DL + vehicle domain)
    ├── video_pipeline.py     # Frame-sampling + temporal analysis
    ├── audio_pipeline.py     # Spectral feature analysis + trained AI-gen classifier
    ├── document_pipeline.py  # AI-text classifier + linguistic statistics
    ├── vehicle_ai_gen_classifier.py  # Domain-specific vehicle AI-gen classifier (CLIP embeddings)
    ├── audio_ai_gen_classifier.py    # Domain-specific audio AI-gen classifier (wav2vec2 embeddings)
    ├── face_ai_gen_classifier.py     # Domain-specific face AI-gen classifier (CLIP embeddings)
    ├── photo_edit_classifier.py      # Domain-specific local-photo-edit classifier (CLIP embeddings)
    └── models/                # Trained .joblib classifiers live here (not in source control until trained)
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Test it (note: `/analyze` requires auth as of v6.0 — see Auth below):

```bash
curl http://localhost:8000/health
```

## Deploy to Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → New → Web Service → connect this GitHub repo.
3. Render auto-detects the `Procfile` for the start command.
4. Set the environment variables below in the Render dashboard (Environment tab).
5. Deploy. Copy the Render URL (e.g. `https://dfd-backend.onrender.com`).
6. In your frontend (Vercel) project's environment variables, set `NEXT_PUBLIC_API_BASE_URL` to that Render URL, and redeploy the frontend.

> **Note (27 Aug 2026):** this repo previously also contained a `railway.json` and Railway-specific instructions in this README, left over from an earlier deployment attempt that was abandoned in favor of Render. `railway.json` has been removed — the actual deployment target is Render, matching `db.py`'s own connection-pool sizing comment and the `Procfile`.

## Environment variables

**Required — the app will fail on startup or first use without these:**

| Variable | Used by | Description |
|---|---|---|
| `DATABASE_URL` | `db.py` | Supabase Postgres connection string (transaction pooler mode) |
| `JWT_SECRET` | `auth.py` | Secret for signing session JWTs |
| `STRIPE_SECRET_KEY` | `billing.py` | Stripe API secret key |
| `STRIPE_WEBHOOK_SECRET` | `billing.py` | Stripe webhook signing secret |
| `BACKEND_URL` | `oauth.py` | This backend's own public URL (used to build OAuth redirect URIs) |
| `FRONTEND_URL` | `oauth.py`, `billing.py` | The deployed frontend's URL (OAuth redirects, Stripe checkout redirects) |

**Optional — have safe defaults or gracefully disable a feature if unset:**

| Variable | Default | Description |
|---|---|---|
| `DFD_TMP_DIR` | system temp dir + `/dfd` | Where uploads and per-job tmp files are stored |
| `DFD_ALLOWED_ORIGINS` | the deployed frontend's URL | Comma-separated list of allowed CORS origins |
| `JWT_EXPIRE_MINUTES` | `60` | Session JWT lifetime |
| `RATE_LIMIT_ANALYZE_PER_MINUTE` | `5/minute` | Per-IP burst limit on `/analyze` |
| `RATE_LIMIT_AUTH_PER_MINUTE` | `10/minute` | Per-IP burst limit on `/auth/*` |
| `STRIPE_PRICE_MONTHLY` | placeholder | Stripe Price ID for the monthly plan — billing checkout won't work correctly until this is set |
| `STRIPE_PRICE_ENTERPRISE` | placeholder | Stripe Price ID for the enterprise plan — same caveat |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | unset | Enables "Continue with Google" — endpoint returns 503 if either is unset |
| `FACEBOOK_APP_ID` / `FACEBOOK_APP_SECRET` | unset | Enables "Continue with Facebook" — same, 503 if unset |
| `PORT` | set automatically by Render | — |

## Key fixes, most recent first

**v6.1 (27 Aug 2026):** free-tier `monthly_quota` was inconsistent — new signups (`auth.py`/`oauth.py`) got `20`, but a canceled subscription reverting to free (`billing.py`) got `200`. Standardized on `200`.

**v6.0:** added auth (JWT + API keys), billing (Stripe), OAuth (Google/Facebook), per-user monthly quotas, real MIME sniffing on upload, async `/analyze` (returns a `job_id`, poll `/jobs/{job_id}`), and domain-specific trained classifiers (CLIP/wav2vec2 embeddings + logistic regression) for vehicle, audio, and face AI-generation detection, replacing or supplementing generic pretrained detectors that showed generator-specific accuracy collapse on this content.

**v5.1:** `prithivMLmods/Deep-Fake-Detector-v2-Model` had a confirmed inverted label mapping in its HuggingFace config — `{'label': 'Deepfake', 'score': X}` actually meant the model computed confidence X for *Realism* (real), not Deepfake. Corrected in `detector/helpers.py:extract_fake_score()` by inverting the score. Validated via batch testing on 14 labeled images from the Kaggle deepfake-and-real-images dataset: 92.9% overall accuracy after the fix (vs. 6.7% before).
