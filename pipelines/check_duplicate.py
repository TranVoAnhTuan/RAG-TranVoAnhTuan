import hashlib
import logging

from app.db.minio_client import minio_client
from app.db.sqlite_db import sqlite_db

from .state import IngestionState

logger = logging.getLogger(__name__)


def get_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def check_duplicate_node(state: IngestionState) -> dict:
    logger.info("🔍 Checking for duplicate documents...")
    file_path = state["file_path"]
    filename = state.get("filename", "unknown.pdf")

    file_hash = get_file_hash(file_path)

    with sqlite_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT minio_url FROM documents_metadata WHERE file_hash = ?", (file_hash,))
        existing_doc = cursor.fetchone()

    if existing_doc:
        logger.warning(f"⚠️ Document already exists (Hash: {file_hash}). Skipping remaining steps.")
        return {
            "file_hash": file_hash,
            "is_duplicate": True,
            "status": "Skipped: Document already exists in database.",
            "minio_url": existing_doc[0],
        }

    topic = state.get("topic", "General")

    logger.info("🆕 New document detected. Saving to MinIO...")
    object_name = f"{file_hash}_{filename}"
    minio_url = minio_client.upload_file(file_path, object_name)

    with sqlite_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents_metadata (file_hash, filename, minio_url, topic) VALUES (?, ?, ?, ?)",
            (file_hash, filename, minio_url, topic),
        )
        conn.commit()

    return {
        "file_hash": file_hash,
        "is_duplicate": False,
        "status": "Saved to MinIO and SQLite. Proceeding with pipeline.",
        "minio_url": minio_url,
    }
