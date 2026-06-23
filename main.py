"""
Deepfake Detection API — FastAPI application v5.2
Key change from v5.1: graphs are no longer base64-encoded in /analyze response.
Instead, graph images are served via GET /graph/{job_id}/{filename}.
This drops response size from ~3.6MB to ~5KB, fixing intermittent 502s on Render.
"""
import os
import re
import hashlib
from datetime import datetime
import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from detector.config import TMP_FOLDER, REPORT_FOLDER, MAX_UPLOAD_BYTES, ALLOWED_ORIGINS
from detector.pipeline import run_pipeline

app = FastAPI(title='Deepfake Detection API', version='5.2')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

def mem():
    process = psutil.Process()
    print(
        f"MEMORY: {process.memory_info().rss / 1024 / 1024:.1f} MB"
    )

@app.get('/health')
async def health():
    import torch
    return {
        'status' : 'ok',
        'version': '5.2',
        'gpu'    : torch.cuda.is_available(),
    }


@app.post('/analyze')
async def analyze_file(file: UploadFile = File(...)):
    """
    Main analysis endpoint.
    Returns lightweight JSON (~5KB) — graphs are fetched separately via /graph/.
    """
    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'File too large (max 500MB)')

    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', file.filename or 'upload')
    job_id    = hashlib.md5(f'{safe_name}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]
    tmp_path  = os.path.join(TMP_FOLDER, f'{job_id}_{safe_name}')

    with open(tmp_path, 'wb') as f:
        f.write(contents)

    print(f'\n[{job_id}] Received: {safe_name} ({len(contents):,} bytes)')

    try:
      print("=== START PIPELINE ===")
      mem()

      result = run_pipeline(tmp_path, job_id)

      print("=== END PIPELINE ===")
      mem()

finally:
        try: os.remove(tmp_path)
        except Exception: pass

    response_kb = len(str(result)) / 1024
    print(f'[{job_id}] Response size: {response_kb:.1f} KB  |  graphs: {len(result.get("graphs", []))}')

    return JSONResponse(content=result)


@app.get('/graph/{job_id}/{filename}')
async def get_graph(job_id: str, filename: str):
    """
    Serve a graph image for a completed job.
    Frontend fetches each graph individually after receiving the /analyze response.
    Graph files are kept in TMP_FOLDER/{job_id}/ until the next analysis clears them.
    """
    safe_id   = re.sub(r'[^A-Za-z0-9]', '', job_id)
    safe_file = re.sub(r'[^A-Za-z0-9._-]', '', filename)
    path      = os.path.join(TMP_FOLDER, safe_id, safe_file)
    if not os.path.exists(path):
        raise HTTPException(404, 'Graph not found or expired')
    return FileResponse(path, media_type='image/png')


@app.get('/report/{job_id}')
async def download_report(job_id: str):
    """
    Download the PDF report for a completed job.
    """
    safe_id = re.sub(r'[^A-Za-z0-9]', '', job_id)
    path    = os.path.join(REPORT_FOLDER, f'report_{safe_id}.pdf')
    if not os.path.exists(path):
        raise HTTPException(404, 'Report not found or expired')
    return FileResponse(
        path,
        media_type='application/pdf',
        filename=f'deepfake_report_{safe_id}.pdf',
    )
