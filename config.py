"""Centralized environment-driven configuration for the support bridge."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Google Gen AI ---
    google_api_key: str
    gemini_model: str = "gemini-3.1-pro-preview"
    embedding_model: str = "gemini-embedding-2-preview"
    embedding_dimension: int = 768

    # --- Pinecone ---
    pinecone_api_key: str
    pinecone_index_name: str = "it-support-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "default"
    top_k_results: int = 5
    min_relevance_score: float = 0.72

    # --- Jira ---
    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_project_key: str = "SUP"
    jira_issue_type: str = "Bug"

    # --- Slack ---
    slack_signing_secret: str
    slack_bot_token: str
    slack_bot_user_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
