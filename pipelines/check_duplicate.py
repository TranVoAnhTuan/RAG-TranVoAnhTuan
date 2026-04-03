import hashlib
from .state import IngestionState
from app.db.sqlite_db import sqlite_db
from app.db.minio_client import minio_client

def check_duplicate_node(state: IngestionState):
    print("🔍 Checking for duplicate documents...")
    raw_text = state["raw_text"]
    file_path = state["file_path"]
    filename = state.get("filename", "unknown.pdf")

    # 1. Extract the first 5 and last 5 lines
    lines = raw_text.split('\n')
    first_5 = lines[:5]
    last_5 = lines[-5:] if len(lines) >= 5 else lines
    content_to_hash = '\n'.join(first_5 + last_5)

    # 2. Create SHA-256 hash
    hash_obj = hashlib.sha256(content_to_hash.encode('utf-8'))
    file_hash = hash_obj.hexdigest()

    # 3. Check existence in SQLite
    with sqlite_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT minio_url FROM documents_metadata WHERE file_hash = ?", (file_hash,))
        existing_doc = cursor.fetchone()

    if existing_doc:
        print(f"⚠️ Document already exists (Hash: {file_hash}). Skipping remaining steps.")
        return {
            "file_hash": file_hash,
            "is_duplicate": True,
            "status": "Skipped: Document already exists in database.",
            "minio_url": existing_doc[0]
        }
    
    # 4. If new, save file to MinIO
    print("🆕 New document detected. Saving to MinIO...")
    object_name = f"{file_hash}_{filename}"
    minio_url = minio_client.upload_file(file_path, object_name)

    # 5. Save metadata to SQLite
    with sqlite_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents_metadata (file_hash, filename, minio_url) VALUES (?, ?, ?)",
            (file_hash, filename, minio_url)
        )
        conn.commit()

    return {
        "file_hash": file_hash,
        "is_duplicate": False,
        "status": "Saved to MinIO and SQLite. Proceeding with pipeline.",
        "minio_url": minio_url
    }