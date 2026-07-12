"""FastAPI entrypoint for the Agentic Enterprise Support Bridge.

Exposes a single Slack Events API webhook. Slack event payload handling
follows https://api.slack.com/events-api: URL verification challenges must
be echoed back verbatim, and every request must be authenticated via the
HMAC-signed `X-Slack-Signature` header before any payload is trusted.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agent import agent
from config import settings
from dependencies import app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_TIMESTAMP_TOLERANCE_SECONDS = 60 * 5
BOT_MENTION_PATTERN = re.compile(r"<@[A-Z0-9]+>\s*")
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await app_state.startup()
    logger.info("Support bridge started: Pinecone + Gen AI clients initialized.")
    yield
    await app_state.shutdown()


app = FastAPI(title="Agentic Enterprise Support Bridge", lifespan=lifespan)


def verify_slack_signature(raw_body: bytes, timestamp: str | None, signature: str | None) -> None:
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Missing Slack signature headers.")

    try:
        request_time = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Slack timestamp header.")

    if abs(time.time() - request_time) > SLACK_TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(status_code=401, detail="Slack request timestamp too old (possible replay).")

    sig_basestring = f"v0:{timestamp}:{raw_body.decode('utf-8')}".encode("utf-8")
    computed_signature = "v0=" + hmac.new(
        settings.slack_signing_secret.encode("utf-8"),
        sig_basestring,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(computed_signature, signature):
        raise HTTPException(status_code=401, detail="Invalid Slack signature.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def post_slack_message(channel: str, text: str, thread_ts: str | None) -> None:
    http_client = app_state.get_deps().http_client
    response = await http_client.post(
        SLACK_POST_MESSAGE_URL,
        json={"channel": channel, "text": text, "thread_ts": thread_ts},
        headers={"Authorization": f"Bearer {settings.slack_bot_token}"},
    )
    body = response.json()
    if not body.get("ok"):
        logger.error("Slack chat.postMessage failed for channel=%s: %s", channel, body.get("error"))


async def handle_app_mention(event: dict) -> None:
    user_text = BOT_MENTION_PATTERN.sub("", event.get("text", "")).strip()
    channel = event.get("channel", "unknown")
    thread_ts = event.get("thread_ts") or event.get("ts")

    if not user_text:
        await post_slack_message(
            channel,
            "I didn't catch a question in that mention -- could you rephrase it?",
            thread_ts,
        )
        return

    deps = app_state.get_deps()

    try:
        result = await agent.run(user_text, deps=deps)
    except Exception:
        logger.exception("Agent run failed for channel=%s text=%r", channel, user_text)
        await post_slack_message(
            channel,
            "Something went wrong while processing your request. Please try again.",
            thread_ts,
        )
        return

    await post_slack_message(channel, result.output, thread_ts)


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    raw_body = await request.body()
    verify_slack_signature(
        raw_body,
        request.headers.get("X-Slack-Request-Timestamp"),
        request.headers.get("X-Slack-Signature"),
    )

    payload = await request.json()

    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload.get("challenge", "")})

    # Slack retries delivery on timeout/non-2xx; avoid re-running the agent
    # (and re-filing duplicate Jira tickets) for a retried delivery.
    if request.headers.get("X-Slack-Retry-Num") is not None:
        return JSONResponse({"ok": True})

    if payload.get("type") != "event_callback":
        return JSONResponse({"ok": True})

    event = payload.get("event", {})

    if event.get("type") == "app_mention":
        # Ack Slack immediately (it expects a 200 within 3s) and do the actual
        # work -- agent run + chat.postMessage -- in the background.
        background_tasks.add_task(handle_app_mention, event)

    return JSONResponse({"ok": True})
