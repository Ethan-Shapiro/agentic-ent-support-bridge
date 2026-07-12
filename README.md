# Agentic Enterprise Support Bridge

A Slack bot that answers employee IT support questions from an internal knowledge base, and automatically escalates anything it can't answer confidently to Jira.

Built with [Pydantic AI](https://ai.pydantic.dev/) (Gemini), [Pinecone](https://www.pinecone.io/) for retrieval, and FastAPI for the Slack Events API webhook.

## How it works

```
Slack @mention
      │
      ▼
FastAPI webhook (/slack/events)
  - verifies the HMAC signature on every request
  - acks Slack immediately, then processes in the background
      │
      ▼
Pydantic AI agent (Gemini)
  - MUST search the knowledge base before answering
  - never answers from its own general knowledge
      │
      ├── search_internal_docs ──▶ embed query (Gemini) ──▶ Pinecone similarity search
      │                                                          │
      │                                        relevant docs found? ──▶ answer, grounded in the docs
      │                                                          │
      │                                              no good match ──▶ escalate_to_jira
      │
      └── escalate_to_jira ──▶ creates a real Jira issue (project/priority/description)
      │
      ▼
Reply posted back to the Slack thread via chat.postMessage
```

The agent has exactly two tools, and hard rules in its system prompt ([agent.py](agent.py)) forbid it from answering technical questions out of its own training data — every factual claim has to come from a `search_internal_docs` call, or the question gets escalated to a human via Jira instead.

## Setup

1. **Python env**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment variables** — copy `.env.example` to `.env` and fill in:
   - `GOOGLE_API_KEY` — Gemini API key
   - `PINECONE_API_KEY` + index/cloud/region config
   - `JIRA_BASE_URL` / `JIRA_EMAIL` / `JIRA_API_TOKEN` / `JIRA_PROJECT_KEY` / `JIRA_ISSUE_TYPE`
   - `SLACK_SIGNING_SECRET` / `SLACK_BOT_TOKEN` / `SLACK_BOT_USER_ID`

3. **Seed the knowledge base** — embeds `rag_dataset.csv` and upserts it into a Pinecone serverless index (creating the index if it doesn't exist yet):
   ```bash
   python seed_db.py
   ```
   The script is resumable — if it gets interrupted (e.g. by a rate limit), re-running it skips rows that were already embedded.

4. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

5. **Expose it to Slack** — Slack's Events API needs a public HTTPS URL. For local testing, tunnel port 8000 with [ngrok](https://ngrok.com/):
   ```bash
   ngrok http 8000
   ```
   Then set your Slack app's **Event Subscriptions** Request URL to `https://<your-ngrok-domain>/slack/events` and subscribe to the `app_mention` bot event.

6. Test it without Slack at all:
   ```bash
   python test_workflow.py "How do I reset my IP address?"
   ```

## Slack app configuration

Bot Token Scopes needed:
- `app_mentions:read` — to see @-mentions
- `chat:write` — to post replies

Event Subscriptions: subscribe to `app_mention`.

## Project layout

| File | Purpose |
|---|---|
| [main.py](main.py) | FastAPI app, Slack signature verification, webhook handler |
| [agent.py](agent.py) | The Pydantic AI agent, system prompt, and its two tools |
| [dependencies.py](dependencies.py) | Process-lifetime clients (Pinecone, Gemini, httpx) injected into the agent |
| [config.py](config.py) | Centralized settings loaded from `.env` |
| [seed_db.py](seed_db.py) | One-off script to embed and upsert the knowledge base into Pinecone |
| [test_workflow.py](test_workflow.py) | Runs the agent directly, bypassing Slack, for local testing |
