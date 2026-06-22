# Deepfake Detection API — Backend

FastAPI-based deepfake detection backend. Supports image, video, audio, and document analysis.

## Project structure

```
dfd-backend/
├── main.py                  # FastAPI app — endpoints /health /analyze /report/{id}
├── requirements.txt
├── Procfile                 # Railway start command
├── railway.json
└── detector/
    ├── __init__.py
    ├── config.py            # Folder paths and constants
    ├── result.py            # AnalysisResult class (builds JSON + PDF in parallel)
    ├── helpers.py           # Shared utilities: face detection, label scoring, graph style
    ├── pipeline.py          # Main dispatcher — routes file to correct analysis function
    ├── image_pipeline.py    # 3-stage image analysis (frequency + face forensics + DL)
    ├── video_pipeline.py    # Frame-sampling + temporal analysis
    ├── audio_pipeline.py    # MFCC + spectral feature analysis
    └── document_pipeline.py # AI-text classifier + linguistic statistics
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze -F "file=@your_image.jpg"
```

## Deploy to Railway

1. Create a new GitHub repo and push this folder to it
2. Go to railway.app → New Project → Deploy from GitHub repo → select this repo
3. Railway auto-detects the Procfile and deploys
4. Set environment variable in Railway dashboard:
   - `DFD_ALLOWED_ORIGINS` = your Vercel frontend URL (e.g. `https://veritas-deepfake-detector.vercel.app`)
5. Copy the Railway URL (e.g. `https://dfd-backend-production.up.railway.app`)
6. Go to Vercel → your frontend project → Settings → Environment Variables
7. Update `NEXT_PUBLIC_API_BASE_URL` to the Railway URL
8. Redeploy the frontend on Vercel

Your frontend now points at a permanent backend — no more Colab/ngrok.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DFD_TMP_DIR` | `/tmp/dfd` | Where uploads and per-job tmp files are stored |
| `DFD_ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins. Set to your frontend URL in production. |
| `PORT` | 8000 | Set automatically by Railway |

## Key fix in this version (v5.1)

`prithivMLmods/Deep-Fake-Detector-v2-Model` has a confirmed inverted label mapping
in its HuggingFace config — `{'label': 'Deepfake', 'score': X}` actually means
the model computed confidence X for *Realism* (real), not Deepfake.

This is corrected in `detector/helpers.py:extract_fake_score()` by inverting the score:
`return (1 - label_map[key]) * 100, key`

Validated via batch testing on 14 labeled images from the Kaggle deepfake-and-real-images
dataset: 92.9% overall accuracy after fix (vs 6.7% before).
