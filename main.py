"""
Deepfake Detection API — FastAPI application.
Replaces notebook Cells 14 + 15 (without ngrok, nest_asyncio, or any Colab dependency).
Run with:  uvicorn main:app --host 0.0.0.0 --port 8000
Railway will start this automatically via the Procfile.
"""
import os
import re
import hashlib
from datetime import datetime
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from detector.config import TMP_FOLDER, REPORT_FOLDER, MAX_UPLOAD_BYTES, ALLOWED_ORIGINS
from detector.pipeline import run_pipeline

app = FastAPI(title='Deepfake Detection API', version='5.1')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
async def health():
    import torch
    return {
        'status' : 'ok',
        'version': '5.1',
        'gpu'    : torch.cuda.is_available(),
    }


@app.post('/analyze')
async def analyze_file(file: UploadFile = File(...)):
    """
    Main endpoint.
    Frontend POSTs the file here as multipart/form-data.
    Returns JSON with results + base64-encoded graphs.
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
        result = run_pipeline(tmp_path, job_id)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

    
    response_size = len(json.dumps(result))
    print(f"RESPONSE SIZE = {response_size/1024:.2f} KB")
    return JSONResponse(content=result)


@app.get('/report/{job_id}')
async def download_report(job_id: str):
    """
    Download the PDF report for a completed job.
    Frontend hits this URL when user clicks 'Download PDF'.
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
