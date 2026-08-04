"""
billing.py — Stripe (test mode) subscription checkout + webhook handling.

Skeletal by design: two plans (monthly, enterprise), no proration/upgrade-downgrade
logic, no invoice history endpoint. Flip STRIPE_SECRET_KEY to a live key when ready
to charge real money — nothing else changes.
"""
import os

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request

from db import get_pool
from auth import get_current_user

stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
STRIPE_WEBHOOK_SECRET = os.environ["STRIPE_WEBHOOK_SECRET"]

router = APIRouter(prefix="/billing", tags=["billing"])


def stripe_get(obj, key, default=None):
    """
    Safe accessor for Stripe SDK objects (StripeObject and subclasses).

    Confirmed via live traceback (billing.py:99, 3 Aug 2026) that these
    objects do NOT implement .get() the way a dict does -- unknown
    attribute access (including "get") is routed through __getattr__
    and raises AttributeError instead of returning a bound method:

        AttributeError: get

    Bracket access (obj[key]) IS confirmed working elsewhere in this file
    (session["metadata"]["internal_user_id"], subscription["id"]), but
    raises KeyError when the key is absent. This wraps that in a
    dict.get()-equivalent so optional fields don't crash the webhook.
    """
    try:
        return obj[key]
    except (KeyError, TypeError):
        return default

PLAN_CONFIG = {
    "monthly": {"price_id": os.environ.get("STRIPE_PRICE_MONTHLY", "price_REPLACE_ME"), "quota": 500},
    "enterprise": {"price_id": os.environ.get("STRIPE_PRICE_ENTERPRISE", "price_REPLACE_ME"), "quota": 999999},
}


@router.post("/checkout-session")
async def create_checkout_session(plan: str, user: dict = Depends(get_current_user)):
    if plan not in PLAN_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown plan '{plan}'")

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=user["email"],
        line_items=[{"price": PLAN_CONFIG[plan]["price_id"], "quantity": 1}],
        # NOTE: the frontend is a single-page app (just app/page.js) with no separate
        # /billing/success or /billing/cancel routes, so we redirect back to root with
        # a query param instead — page.js reads ?billing=success|cancel on load.
        success_url=os.environ["FRONTEND_URL"] + "/?billing=success",
        cancel_url=os.environ["FRONTEND_URL"] + "/?billing=cancel",
        client_reference_id=user["id"],
        metadata={"internal_user_id": user["id"], "plan": plan},
    )
    return {"checkout_url": session.url}


@router.get("/subscription")
async def get_subscription(user: dict = Depends(get_current_user)):
    """Current plan + usage, for the frontend to display (e.g. 'Free — 3/20 used this month')."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        sub = await conn.fetchrow(
            "SELECT plan, status, monthly_quota, current_period_start FROM subscriptions WHERE user_id = $1",
            user["id"],
        )
        if sub is None:
            raise HTTPException(status_code=404, detail="No subscription found")

        used = await conn.fetchval(
            "SELECT count(*) FROM usage_records WHERE user_id = $1 AND created_at >= $2",
            user["id"], sub["current_period_start"],
        )

    return {
        "plan": sub["plan"],
        "status": sub["status"],
        "monthly_quota": sub["monthly_quota"],
        "used_this_period": used,
    }


@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    pool = await get_pool()

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"]["internal_user_id"]
        plan = session["metadata"]["plan"]
        quota = PLAN_CONFIG[plan]["quota"]

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE subscriptions
                SET plan = $1, monthly_quota = $2, status = 'active',
                    stripe_customer_id = $3, stripe_subscription_id = $4,
                    current_period_start = now(), updated_at = now()
                WHERE user_id = $5
                """,
                plan, quota, stripe_get(session, "customer"), stripe_get(session, "subscription"), user_id,
            )

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE subscriptions SET plan = 'free', monthly_quota = 200, status = 'canceled', updated_at = now()
                WHERE stripe_subscription_id = $1
                """,
                subscription["id"],
            )

    elif event["type"] == "invoice.payment_failed":
        subscription_id = stripe_get(event["data"]["object"], "subscription")
        if subscription_id:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE subscriptions SET status = 'past_due', updated_at = now()
                    WHERE stripe_subscription_id = $1
                    """,
                    subscription_id,
                )

    return {"received": True}
