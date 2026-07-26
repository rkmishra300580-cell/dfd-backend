"""
db.py — Supabase Postgres connection pool (asyncpg).

Uses Supabase's "Transaction" pooler mode (PgBouncer). That mode does not support
asyncpg's server-side prepared-statement caching, so statement_cache_size=0 is
required — do not remove it, queries will intermittently fail under load without it.
"""
import os
import json
from typing import Optional

import asyncpg

DATABASE_URL = os.environ["DATABASE_URL"]

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection):
    # Auto encode/decode jsonb <-> Python dict, so callers can pass/receive plain dicts.
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def init_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=5,  # Render Standard has limited RAM/CPU — keep this small
            command_timeout=30,
            statement_cache_size=0,  # required for Supabase's pgbouncer transaction mode
            init=_init_connection,
        )
    return _pool


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        return await init_db_pool()
    return _pool


async def close_db_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
