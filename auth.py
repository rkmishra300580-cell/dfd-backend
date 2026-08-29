"""
auth.py — email/password signup+login (JWT sessions) and API-key management.

Two auth methods, both accepted by get_current_user():
  Authorization: Bearer <jwt>              -> browser/session use, from /auth/login
  Authorization: ApiKey <raw_api_key>      -> programmatic use, from /auth/api-keys

API keys are stored only as a sha256 hash. The raw key is shown to the user exactly
once, at creation time, and cannot be retrieved again.
"""
import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, EmailStr

from db import get_pool

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))


# ---------- password hashing ----------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ---------- JWT ----------

def create_access_token(user_id: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------- API keys ----------

def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key_to_show_user_once, key_hash_to_store, key_prefix_for_display)."""
    raw = "dpk_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    prefix = raw[:12]
    return raw, key_hash, prefix


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------- request/response models ----------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ApiKeyResponse(BaseModel):
    api_key: str
    prefix: str
    warning: str = "Store this key now — it will not be shown again."


# ---------- current-user dependency ----------

async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, credential = authorization.partition(" ")
    pool = await get_pool()

    if scheme.lower() == "bearer":
        payload = decode_access_token(credential)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, email, role FROM users WHERE id = $1", payload["sub"]
            )
        if row is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {"id": str(row["id"]), "email": row["email"], "role": row["role"], "auth_method": "jwt"}

    elif scheme.lower() == "apikey":
        key_hash = hash_api_key(credential)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.id, u.email, u.role
                FROM api_keys k JOIN users u ON u.id = k.user_id
                WHERE k.key_hash = $1 AND k.revoked_at IS NULL
                """,
                key_hash,
            )
        if row is None:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        return {"id": str(row["id"]), "email": row["email"], "role": row["role"], "auth_method": "apikey"}

    raise HTTPException(status_code=401, detail="Unsupported authorization scheme")


def require_role(*allowed_roles: str):
    async def dependency(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user
    return dependency


# ---------- routes ----------

@router.post("/signup", response_model=TokenResponse)
async def signup(body: SignupRequest):
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", body.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        password_hash = hash_password(body.password)
        row = await conn.fetchrow(
            """
            INSERT INTO users (email, password_hash, role)
            VALUES ($1, $2, 'analyst')
            RETURNING id, role
            """,
            body.email, password_hash,
        )
        await conn.execute(
            # BUG FIX (27 Aug 2026): was 20 - inconsistent with billing.py's
            # subscription-cancellation handler, which reverts a canceled
            # subscription to 'free' with monthly_quota=200. Same plan tier
            # was getting two different quotas depending on which code path
            # assigned it - confirmed 200 is the correct value, this was the
            # stale one.
            "INSERT INTO subscriptions (user_id, plan, monthly_quota) VALUES ($1, 'free', 200)",
            row["id"],
        )

    token = create_access_token(str(row["id"]), row["role"])
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, password_hash, role FROM users WHERE email = $1", body.email
        )
    # row["password_hash"] can be NULL for OAuth-only accounts (see oauth.py) —
    # check that explicitly, since bcrypt.checkpw() would raise on None rather
    # than just failing the comparison.
    if row is None or row["password_hash"] is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(row["id"]), row["role"])
    return TokenResponse(access_token=token)


@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(label: str = "default", user: dict = Depends(get_current_user)):
    raw, key_hash, prefix = generate_api_key()
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, label) VALUES ($1, $2, $3, $4)",
            user["id"], key_hash, prefix, label,
        )
    return ApiKeyResponse(api_key=raw, prefix=prefix)


@router.delete("/api-keys/{prefix}")
async def revoke_api_key(prefix: str, user: dict = Depends(get_current_user)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE api_keys SET revoked_at = now()
            WHERE user_id = $1 AND key_prefix = $2 AND revoked_at IS NULL
            """,
            user["id"], prefix,
        )
    if result.endswith(" 0"):
        raise HTTPException(status_code=404, detail="Key not found or already revoked")
    return {"revoked": True}
