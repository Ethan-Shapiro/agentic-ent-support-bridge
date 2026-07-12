"""Runtime dependencies injected into the Pydantic AI agent via RunContext.

Clients are built once per warm container/process and reused across
requests (this app is stateless per-request, not per-process) so that
connection pools and TCP/TLS handshakes are amortized across invocations,
which matters for both Cloud Run/Lambda-style serverless deployments.
"""
from __future__ import annotations

import httpx
from google import genai
from pinecone import Index, Pinecone
from pydantic import BaseModel, ConfigDict

from config import settings


class AgentDeps(BaseModel):
    """Dependencies available to every agent tool via RunContext[AgentDeps]."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pinecone_index: Index
    genai_client: genai.Client
    http_client: httpx.AsyncClient


def build_pinecone_index() -> Index:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.indexes.exists(settings.pinecone_index_name):
        raise RuntimeError(
            f"Pinecone index '{settings.pinecone_index_name}' does not exist. "
            "Run seed_db.py first to create and populate it."
        )
    return pc.index(settings.pinecone_index_name)


def build_genai_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


class AppState:
    """Holds process-lifetime singletons, wired up via FastAPI's lifespan."""

    def __init__(self) -> None:
        self.pinecone_index: Index | None = None
        self.genai_client: genai.Client | None = None
        self.http_client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self.pinecone_index = build_pinecone_index()
        self.genai_client = build_genai_client()
        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def shutdown(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()

    def get_deps(self) -> AgentDeps:
        if not (self.pinecone_index and self.genai_client and self.http_client is not None):
            raise RuntimeError("AppState was not initialized. Did the lifespan startup run?")
        return AgentDeps(
            pinecone_index=self.pinecone_index,
            genai_client=self.genai_client,
            http_client=self.http_client,
        )


app_state = AppState()
