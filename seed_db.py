"""Standalone script: embed rag_dataset.csv and upsert it into Pinecone.

Usage:
    python seed_db.py

Expects a local `rag_dataset.csv` with at least `topic` and `content`
columns. Creates the target Pinecone serverless index if it doesn't already
exist, then embeds and upserts every row in batches.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai.errors import ClientError
from pinecone import Index, Pinecone, ServerlessSpec, Vector

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = Path("rag_dataset.csv")
BATCH_SIZE = 50
EMBED_RETRY_ATTEMPTS = 3
EMBED_RETRY_DELAY_SECONDS = 30


def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        logger.error("Could not find %s in the current directory.", CSV_PATH)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    missing_columns = {"ki_topic", "ki_text"} - set(df.columns)
    if missing_columns:
        logger.error("rag_dataset.csv is missing required column(s): %s", missing_columns)
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=["ki_topic", "ki_text"])
    df = df[df["ki_text"].str.strip() != ""]
    if len(df) < before:
        logger.warning("Dropped %d row(s) with empty ki_topic/ki_text.", before - len(df))

    return df.reset_index(drop=True)


def ensure_index(pc: Pinecone) -> None:
    if pc.indexes.exists(settings.pinecone_index_name):
        logger.info("Pinecone index '%s' already exists.", settings.pinecone_index_name)
        return

    logger.info("Creating Pinecone serverless index '%s'...", settings.pinecone_index_name)
    pc.indexes.create(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )

    while not pc.indexes.describe(settings.pinecone_index_name).status.ready:
        logger.info("Waiting for index to become ready...")
        time.sleep(1)

    logger.info("Index '%s' is ready.", settings.pinecone_index_name)


def existing_ids(index: Index, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        response = index.fetch(ids=chunk, namespace=settings.pinecone_namespace)
        found.update(response.vectors.keys())
    return found


def embed_one(client: genai.Client, text: str) -> list[float]:
    # gemini-embedding-2 collapses a multi-item `contents` list into a single
    # combined embedding instead of one per item, so each text needs its own call.
    for attempt in range(1, EMBED_RETRY_ATTEMPTS + 1):
        try:
            response = client.models.embed_content(
                model=settings.embedding_model,
                contents=[text],
                config={"output_dimensionality": settings.embedding_dimension},
            )
            return response.embeddings[0].values
        except ClientError as exc:
            if exc.code != 429 or attempt == EMBED_RETRY_ATTEMPTS:
                raise
            logger.warning(
                "Rate limited by Gemini API (attempt %d/%d), waiting %ds...",
                attempt,
                EMBED_RETRY_ATTEMPTS,
                EMBED_RETRY_DELAY_SECONDS,
            )
            time.sleep(EMBED_RETRY_DELAY_SECONDS)


def main() -> None:
    df = load_dataset()
    logger.info("Loaded %d rows from %s.", len(df), CSV_PATH)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(pc)
    index = pc.index(settings.pinecone_index_name)

    genai_client = genai.Client(api_key=settings.google_api_key)

    all_ids = [f"doc-{i}" for i in range(len(df))]
    already_done = existing_ids(index, all_ids)
    if already_done:
        logger.info("%d/%d rows already embedded from a previous run; skipping those.", len(already_done), len(df))

    total_upserted = len(already_done)
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        pending = [
            (start + i, row) for i, (_, row) in enumerate(batch.iterrows()) if f"doc-{start + i}" not in already_done
        ]
        if not pending:
            continue

        vectors = [
            Vector(
                id=f"doc-{row_index}",
                values=embed_one(genai_client, str(row["ki_text"])),
                metadata={"ki_topic": str(row["ki_topic"]), "ki_text": str(row["ki_text"])},
            )
            for row_index, row in pending
        ]

        index.upsert(vectors=vectors, namespace=settings.pinecone_namespace)
        total_upserted += len(vectors)
        logger.info("Upserted %d/%d vectors.", total_upserted, len(df))

    logger.info("Done. Upserted %d vectors into '%s'.", total_upserted, settings.pinecone_index_name)


if __name__ == "__main__":
    main()
