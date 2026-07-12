"""The Pydantic AI agent that powers the enterprise IT support bridge.

The agent is intentionally restricted to two tools: searching the internal
knowledge base and escalating to Jira. It must never answer from its own
world knowledge -- IT support answers that aren't grounded in the retrieved
documents are worse than no answer, because they erode trust and can send
someone down the wrong remediation path.
"""
from __future__ import annotations

import base64
import logging
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from config import settings
from dependencies import AgentDeps

logger = logging.getLogger(__name__)

JIRA_PRIORITIES = Literal["Highest", "High", "Medium", "Low", "Lowest"]

SYSTEM_PROMPT = """\
You are the Enterprise IT Support Bridge, a first-line support agent for employees.

Hard rules, no exceptions:
1. You MUST NOT answer any technical or IT support question from your own general
   knowledge. Every factual claim about internal systems, tools, policies, or
   troubleshooting steps must come from the `search_internal_docs` tool.
2. Always call `search_internal_docs` first for any question that could be
   answered by internal documentation, even if you think you already know
   the answer. Your training data may be wrong, outdated, or not reflect this
   company's specific setup.
3. If `search_internal_docs` returns no relevant results, or the results do not
   actually answer the user's question, do NOT guess or improvise an answer.
   Instead call `escalate_to_jira` to create a ticket for a human to follow up,
   and tell the user you've escalated it.
4. If the user reports something that sounds like a bug, an outage, data loss,
   a security concern, or anything they explicitly ask to be escalated, call
   `escalate_to_jira` regardless of what the docs say.
5. Never fabricate ticket numbers, document contents, or system status. Only
   report information that was actually returned by a tool call in this run.
6. Keep responses short, factual, and actionable. State clearly whether you
   answered from documentation or escalated to a human.
"""

google_model = GoogleModel(
    settings.gemini_model,
    provider=GoogleProvider(api_key=settings.google_api_key),
)

agent = Agent(
    google_model,
    deps_type=AgentDeps,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)


@agent.tool
async def search_internal_docs(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the internal IT knowledge base for documentation relevant to the query.

    Args:
        ctx: Run context carrying the Pinecone index and Gen AI client.
        query: The employee's question, rephrased as a focused search query.
    """
    embedding_response = await ctx.deps.genai_client.aio.models.embed_content(
        model=settings.embedding_model,
        contents=[query],
        config={"output_dimensionality": settings.embedding_dimension},
    )
    query_vector = embedding_response.embeddings[0].values

    results = ctx.deps.pinecone_index.query(
        vector=query_vector,
        top_k=settings.top_k_results,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )

    relevant_matches = [
        match for match in results.matches if match.score >= settings.min_relevance_score
    ]

    if not relevant_matches:
        return (
            "No relevant internal documentation was found for this query. "
            "Do not answer from general knowledge -- escalate to Jira instead."
        )

    formatted = []
    for match in relevant_matches:
        topic = match.metadata.get("ki_topic", "unknown")
        content = match.metadata.get("ki_text", "")
        formatted.append(f"[topic: {topic} | relevance: {match.score:.2f}]\n{content}")

    return "\n\n---\n\n".join(formatted)


@agent.tool
async def escalate_to_jira(
    ctx: RunContext[AgentDeps],
    title: str,
    description: str,
    priority: JIRA_PRIORITIES,
) -> str:
    """Create a Jira issue to escalate a bug, outage, or unanswerable question to a human.

    Args:
        ctx: Run context carrying the shared httpx client.
        title: Short, specific summary of the issue (becomes the Jira ticket summary).
        description: Full details of what the user reported, including any relevant
            context gathered during this conversation.
        priority: One of "Highest", "High", "Medium", "Low", "Lowest".
    """
    payload = {
        "fields": {
            "project": {"key": settings.jira_project_key},
            "summary": title,
            "issuetype": {"name": settings.jira_issue_type},
            "priority": {"name": priority},
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}],
                    }
                ],
            },
        }
    }

    auth_token = base64.b64encode(
        f"{settings.jira_email}:{settings.jira_api_token}".encode()
    ).decode()

    try:
        response = await ctx.deps.http_client.post(
            f"{settings.jira_base_url}/rest/api/3/issue",
            json=payload,
            headers={
                "Authorization": f"Basic {auth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
    except Exception:
        logger.exception("Failed to create Jira issue for escalation: %s", title)
        return (
            "I attempted to escalate this to Jira but the request failed. "
            "Please contact IT support directly and reference: " + title
        )

    ticket = response.json()
    ticket_key = ticket.get("key", "UNKNOWN")
    ticket_url = f"{settings.jira_base_url}/browse/{ticket_key}"
    return f"Escalated to Jira as {ticket_key} ({priority} priority): {ticket_url}"
