"""
oauth.py — "Continue with Google" / "Continue with Facebook" (OAuth 2.0
authorization code flow, server-side/"confidential client" style since this
is a plain backend, not a mobile/SPA-only PKCE flow).

Flow for each provider, identical shape:
  1. Frontend does a full page navigation (NOT a fetch — this must be a real
     browser redirect) to GET /auth/{provider}/login
  2. That redirects the browser to the provider's own consent screen
  3. User approves -> provider redirects back to /auth/{provider}/callback?code=...&state=...
  4. Backend exchanges the code for an access token, fetches the user's
     profile, finds-or-creates a local user row, issues our own JWT, and
     redirects to the frontend with the token in a URL FRAGMENT
     (#auth_token=...), not a query param — fragments are never sent to the
     server on subsequent requests and don't appear in server access logs or
     Referer headers, unlike query params.

CSRF protection: a random `state` value is generated on /login and stored in
an in-memory dict with a short TTL, checked on /callback. In-memory only —
same known limitation as jobs.py's BackgroundTasks approach: doesn't survive
a restart and doesn't work across multiple Render instances. Fine at current
scale (single instance); revisit (e.g. store in Postgres) if you ever scale out.

Account linking: if the OAuth email matches an existing user (e.g. they
originally signed up with email+password), the OAuth identity is LINKED to
that existing account rather than creating a duplicate — they can then log
in via either method.
"""
import os
import time
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from db import get_pool
from auth import create_access_token

router = APIRouter(prefix="/auth", tags=["oauth"])

BACKEND_URL = os.environ["BACKEND_URL"]
FRONTEND_URL = os.environ["FRONTEND_URL"]

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
FACEBOOK_APP_ID = os.environ.get("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.environ.get("FACEBOOK_APP_SECRET")

STATE_TTL_SECONDS = 600
_pending_states: dict[str, float] = {}  # state -> expiry epoch seconds


def _new_state() -> str:
    state = secrets.token_urlsafe(24)
    _pending_states[state] = time.time() + STATE_TTL_SECONDS
    return state


def _consume_state(state: str) -> bool:
    """Returns True iff state was issued by us and hasn't expired. Single-use — pops it."""
    expiry = _pending_states.pop(state, None)
    return expiry is not None and expiry > time.time()


async def _find_or_create_oauth_user(provider: str, provider_user_id: str, email: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Already linked to this exact provider identity?
        row = await conn.fetchrow(
            """
            SELECT u.id, u.email, u.role FROM oauth_identities o
            JOIN users u ON u.id = o.user_id
            WHERE o.provider = $1 AND o.provider_user_id = $2
            """,
            provider, provider_user_id,
        )
        if row:
            return {"id": str(row["id"]), "email": row["email"], "role": row["role"]}

        # Existing account (password-based, or a different provider) with the same email? Link it.
        existing_user = await conn.fetchrow("SELECT id, email, role FROM users WHERE email = $1", email)
        if existing_user:
            await conn.execute(
                """
                INSERT INTO oauth_identities (user_id, provider, provider_user_id)
                VALUES ($1, $2, $3) ON CONFLICT (provider, provider_user_id) DO NOTHING
                """,
                existing_user["id"], provider, provider_user_id,
            )
            return {"id": str(existing_user["id"]), "email": existing_user["email"], "role": existing_user["role"]}

        # Brand new user — no password (NULL, see schema_oauth_migration.sql)
        new_user = await conn.fetchrow(
            "INSERT INTO users (email, password_hash, role) VALUES ($1, NULL, 'analyst') RETURNING id, role",
            email,
        )
        await conn.execute(
            "INSERT INTO subscriptions (user_id, plan, monthly_quota) VALUES ($1, 'free', 20)",
            new_user["id"],
        )
        await conn.execute(
            "INSERT INTO oauth_identities (user_id, provider, provider_user_id) VALUES ($1, $2, $3)",
            new_user["id"], provider, provider_user_id,
        )
        return {"id": str(new_user["id"]), "email": email, "role": new_user["role"]}


def _redirect_with_token(token: str, email: str) -> RedirectResponse:
    frag = urlencode({"auth_token": token, "auth_email": email})
    return RedirectResponse(url=f"{FRONTEND_URL}/#{frag}")


# ── Google ───────────────────────────────────────────────────────────────

@router.get("/google/login")
async def google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google login is not configured on this server")
    redirect_uri = f"{BACKEND_URL}/auth/google/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
        "state": _new_state(),
    }
    return RedirectResponse(url=f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


@router.get("/google/callback")
async def google_callback(code: str = None, state: str = None, error: str = None):
    if error:
        return RedirectResponse(url=f"{FRONTEND_URL}/?oauth_error={error}")
    if not code or not state or not _consume_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    redirect_uri = f"{BACKEND_URL}/auth/google/callback"
    async with httpx.AsyncClient(timeout=10) as client:
        token_res = await client.post("https://oauth2.googleapis.com/token", data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        })
        if token_res.status_code != 200:
            print(f"[oauth] Google token exchange failed: status={token_res.status_code} "
                  f"redirect_uri={redirect_uri!r} body={token_res.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Google token exchange failed: {token_res.text}",
            )
        access_token = token_res.json()["access_token"]

        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_res.status_code != 200:
            raise HTTPException(status_code=502, detail="Google profile fetch failed")
        profile = userinfo_res.json()

    if not profile.get("email"):
        raise HTTPException(status_code=400, detail="Google account has no email on file; cannot sign in")

    user = await _find_or_create_oauth_user("google", profile["sub"], profile["email"])
    jwt_token = create_access_token(user["id"], user["role"])
    return _redirect_with_token(jwt_token, user["email"])


# ── Facebook ─────────────────────────────────────────────────────────────

@router.get("/facebook/login")
async def facebook_login():
    if not FACEBOOK_APP_ID:
        raise HTTPException(status_code=503, detail="Facebook login is not configured on this server")
    redirect_uri = f"{BACKEND_URL}/auth/facebook/callback"
    params = {
        "client_id": FACEBOOK_APP_ID,
        "redirect_uri": redirect_uri,
        "scope": "email public_profile",
        "state": _new_state(),
    }
    return RedirectResponse(url=f"https://www.facebook.com/v19.0/dialog/oauth?{urlencode(params)}")


@router.get("/facebook/callback")
async def facebook_callback(code: str = None, state: str = None, error: str = None):
    if error:
        return RedirectResponse(url=f"{FRONTEND_URL}/?oauth_error={error}")
    if not code or not state or not _consume_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    redirect_uri = f"{BACKEND_URL}/auth/facebook/callback"
    async with httpx.AsyncClient(timeout=10) as client:
        token_res = await client.get("https://graph.facebook.com/v19.0/oauth/access_token", params={
            "client_id": FACEBOOK_APP_ID,
            "client_secret": FACEBOOK_APP_SECRET,
            "redirect_uri": redirect_uri,
            "code": code,
        })
        if token_res.status_code != 200:
            raise HTTPException(status_code=502, detail="Facebook token exchange failed")
        access_token = token_res.json()["access_token"]

        profile_res = await client.get("https://graph.facebook.com/me", params={
            "fields": "id,name,email",
            "access_token": access_token,
        })
        if profile_res.status_code != 200:
            raise HTTPException(status_code=502, detail="Facebook profile fetch failed")
        profile = profile_res.json()

    email = profile.get("email")
    if not email:
        # Facebook accounts can lack a verified/available email — rare, but possible.
        raise HTTPException(status_code=400, detail="Facebook account has no email on file; cannot sign in")

    user = await _find_or_create_oauth_user("facebook", profile["id"], email)
    jwt_token = create_access_token(user["id"], user["role"])
    return _redirect_with_token(jwt_token, user["email"])
