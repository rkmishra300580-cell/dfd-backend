"""
jobs.py — async job tracking on top of FastAPI BackgroundTasks + Postgres.

Deliberately NOT Celery/Redis (that costs money to run properly). Known, accepted
limitations of this approach:
  - a job in flight is lost if Render restarts/redeploys mid-analysis
  - single Render instance only — no distributed workers
  - no automatic retries
Revisit with Celery + a free-tier Redis (e.g. Upstash) once real usage volume
justifies the added complexity.
"""
import traceback
from typing import Callable, Any

from fastapi import APIRouter, Depends, HTTPException

from db import get_pool
from auth import get_current_user

router = APIRouter(prefix="/jobs", tags=["jobs"])


async def create_job(job_id: str, user_id: str, modality: str, safe_name: str = None) -> None:
    """job_id is the caller's existing identifier (e.g. the md5-based hash main.py already
    generates for /graph and /report URLs) — not auto-generated here, so those URLs don't change."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO jobs (id, user_id, status, modality, safe_name) VALUES ($1, $2, 'queued', $3, $4)",
            job_id, user_id, modality, safe_name,
        )


async def run_job(job_id: str, work_fn: Callable[..., Any], *args, **kwargs):
    """
    Executes work_fn (awaited — wrap sync/CPU-bound work with asyncio.to_thread
    before passing it in, so it doesn't block the event loop) and records the
    outcome in Postgres. Any cleanup (e.g. deleting a temp file) should be done by
    the caller after awaiting this, not inside work_fn.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE jobs SET status = 'running', started_at = now() WHERE id = $1", job_id)

    try:
        result = await work_fn(*args, **kwargs)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = 'done', result_json = $2, finished_at = now() WHERE id = $1",
                job_id, result,
            )
    except Exception:
        err = traceback.format_exc()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = 'failed', error_text = $2, finished_at = now() WHERE id = $1",
                job_id, err,
            )


@router.get("/{job_id}")
async def get_job_status(job_id: str, user: dict = Depends(get_current_user)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, status, modality, result_json, error_text,
                   created_at, started_at, finished_at
            FROM jobs WHERE id = $1
            """,
            job_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(row["user_id"]) != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not your job")

    return {
        "job_id": str(row["id"]),
        "status": row["status"],
        "modality": row["modality"],
        "result": row["result_json"],
        "error": row["error_text"],
        "created_at": row["created_at"].isoformat(),
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }
