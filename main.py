"""
Deepfake Detection API — FastAPI application v6.0
v5.2 behavior (graphs served via GET /graph/{job_id}/{filename}, not base64) is preserved.

New in v6.0:
  - Auth: signup/login (JWT) + API keys, via /auth/*
  - Billing: Stripe checkout + webhook, via /billing/*
  - Rate limiting: per-IP burst limit (slowapi) + per-user monthly quota (Postgres)
  - Real MIME sniffing on upload (not just trusting the file extension)
  - /analyze is now ASYNCHRONOUS: returns a job_id immediately, poll GET /jobs/{job_id}
  - /graph and /report are now auth-gated to the job's owner (or an admin) — this is a
    behavior change from v5.2, where job_id alone was sufficient. The frontend must now
    send the Authorization header on these two calls as well.
"""
import os
import re
import hashlib
import asyncio
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from detector.config import TMP_FOLDER, REPORT_FOLDER, MAX_UPLOAD_BYTES, ALLOWED_ORIGINS
from detector.pipeline import run_pipeline

from db import init_db_pool, close_db_pool, get_pool
from auth import router as auth_router, get_current_user
from oauth import router as oauth_router
from billing import router as billing_router
from jobs import router as jobs_router, create_job, run_job
from rate_limit import limiter, enforce_monthly_quota, record_usage, ANALYZE_PER_MINUTE
from uploads_guard import validate_upload

app = FastAPI(title='Deepfake Detection API', version='6.0')

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(auth_router)
app.include_router(oauth_router)
app.include_router(billing_router)
app.include_router(jobs_router)


@app.on_event('startup')
async def on_startup():
    await init_db_pool()


@app.on_event('shutdown')
async def on_shutdown():
    await close_db_pool()


@app.get('/health')
async def health():
    import torch
    return {
        'status' : 'ok',
        'version': '6.0',
        'gpu'    : torch.cuda.is_available(),
    }


async def _run_pipeline_and_cleanup(tmp_path: str, job_id: str):
    """
    Runs the existing synchronous, CPU/GPU-bound run_pipeline() in a worker thread
    (so it doesn't block the event loop for other requests), then always removes
    the temp upload file, whether the pipeline succeeded or raised.
    """
    try:
        result = await asyncio.to_thread(run_pipeline, tmp_path, job_id)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post('/analyze')
@limiter.limit(ANALYZE_PER_MINUTE)
async def analyze_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(enforce_monthly_quota),
):
    """
    Main analysis endpoint. CHANGED FROM v5.2: now asynchronous.
    Returns {job_id, status: 'queued'} immediately instead of the full result.
    Poll GET /jobs/{job_id} for status; once status == 'done', result_json holds
    exactly the same shape v5.2 used to return directly (including the graphs list).
    """
    contents = await validate_upload(file, MAX_UPLOAD_BYTES, request)

    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', file.filename or 'upload')
    job_id    = hashlib.md5(f'{safe_name}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]
    tmp_path  = os.path.join(TMP_FOLDER, f'{job_id}_{safe_name}')

    with open(tmp_path, 'wb') as f:
        f.write(contents)

    print(f'\n[{job_id}] Received: {safe_name} ({len(contents):,} bytes)  |  user: {user["email"]}')

    await create_job(job_id, user['id'], modality='image', safe_name=safe_name)
    await record_usage(user['id'], job_id, '/analyze', 'image')

    background_tasks.add_task(run_job, job_id, _run_pipeline_and_cleanup, tmp_path, job_id)

    return JSONResponse(content={
        'job_id': job_id,
        'status': 'queued',
        'check_status_at': f'/jobs/{job_id}',
    })


async def _check_job_ownership(job_id: str, user: dict):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow('SELECT user_id FROM jobs WHERE id = $1', job_id)
    if row is None:
        raise HTTPException(404, 'Job not found')
    if str(row['user_id']) != user['id'] and user['role'] != 'admin':
        raise HTTPException(403, 'Not your job')


@app.get('/graph/{job_id}/{filename}')
async def get_graph(job_id: str, filename: str, user: dict = Depends(get_current_user)):
    """
    Serve a graph image for a completed job.
    CHANGED FROM v5.2: now requires auth + ownership check — job_id alone is no
    longer sufficient to fetch a graph. Frontend must send Authorization header.
    """
    await _check_job_ownership(job_id, user)

    safe_id   = re.sub(r'[^A-Za-z0-9]', '', job_id)
    safe_file = re.sub(r'[^A-Za-z0-9._-]', '', filename)
    path      = os.path.join(TMP_FOLDER, safe_id, safe_file)
    if not os.path.exists(path):
        raise HTTPException(404, 'Graph not found or expired')
    return FileResponse(path, media_type='image/png')


@app.get('/report/{job_id}')
async def download_report(job_id: str, user: dict = Depends(get_current_user)):
    """
    Download the PDF report for a completed job.
    CHANGED FROM v5.2: now requires auth + ownership check, same reasoning as /graph above.
    """
    await _check_job_ownership(job_id, user)

    safe_id = re.sub(r'[^A-Za-z0-9]', '', job_id)
    path    = os.path.join(REPORT_FOLDER, f'{safe_id}.pdf')
    if not os.path.exists(path):
        raise HTTPException(404, 'Report not found or expired')
    return FileResponse(
        path,
        media_type='application/pdf',
        filename=f'deepfake_report_{safe_id}.pdf',
    )
