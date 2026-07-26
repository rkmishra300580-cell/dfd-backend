"""
rate_limit.py — two independent layers:
  1. slowapi: per-IP burst limiting (in-memory, no Redis needed at this scale).
     Protects against raw flooding before auth even matters.
  2. enforce_monthly_quota: per-user quota check backed by Postgres.
     Protects against a legitimate, authenticated user going over their plan's quota.
"""
import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Depends, HTTPException

from db import get_pool
from auth import get_current_user

limiter = Limiter(key_func=get_remote_address)

ANALYZE_PER_MINUTE = os.environ.get("RATE_LIMIT_ANALYZE_PER_MINUTE", "5/minute")
AUTH_PER_MINUTE = os.environ.get("RATE_LIMIT_AUTH_PER_MINUTE", "10/minute")


async def enforce_monthly_quota(user: dict = Depends(get_current_user)) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        sub = await conn.fetchrow(
            """
            SELECT plan, monthly_quota, current_period_start
            FROM subscriptions WHERE user_id = $1
            """,
            user["id"],
        )
        if sub is None:
            raise HTTPException(status_code=403, detail="No active subscription")

        used = await conn.fetchval(
            "SELECT count(*) FROM usage_records WHERE user_id = $1 AND created_at >= $2",
            user["id"], sub["current_period_start"],
        )

    if sub["plan"] != "enterprise" and used >= sub["monthly_quota"]:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly quota exceeded ({used}/{sub['monthly_quota']}). Upgrade your plan to continue.",
        )
    return user


async def record_usage(user_id: str, job_id: str, endpoint: str, modality: str = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO usage_records (user_id, job_id, endpoint, modality) VALUES ($1, $2, $3, $4)",
            user_id, job_id, endpoint, modality,
        )
