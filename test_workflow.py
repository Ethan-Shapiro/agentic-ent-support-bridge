"""Manual smoke test: runs the agent directly against real Pinecone/Jira,
bypassing Slack, so the RAG + escalation pipeline can be validated end to end.

Usage:
    python test_workflow.py "your question here"
"""
from __future__ import annotations

import asyncio
import sys

from agent import agent
from dependencies import app_state


async def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I set up my mobile device for company email?"

    await app_state.startup()
    try:
        deps = app_state.get_deps()
        result = await agent.run(query, deps=deps)
        print(f"\nQuery: {query}\n\nResponse:\n{result.output}")
    finally:
        await app_state.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
