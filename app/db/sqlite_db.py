"""SQLite metadata store for uploaded documents.

Singleton pattern: use the module-level ``sqlite_db`` instance.
"""

import contextlib
import sqlite3

from app.core.config import settings


class SQLiteDB:
    """Tracks document metadata (hash, filename, MinIO URL, topic) in SQLite."""

    def __init__(self) -> None:
        self.db_path: str = settings.SQLITE_DB_PATH
        self._create_table()

    def get_connection(self) -> sqlite3.Connection:
        """Return a new connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        """Ensure the documents_metadata table exists (with topic column)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents_metadata (
                    file_hash TEXT PRIMARY KEY,
                    filename TEXT,
                    minio_url TEXT,
                    topic TEXT
                )
            """)
            with contextlib.suppress(sqlite3.OperationalError):
                cursor.execute("ALTER TABLE documents_metadata ADD COLUMN topic TEXT DEFAULT 'General'")
            conn.commit()

    def get_all_topics(self) -> list[str]:
        """Return distinct topic values from stored documents."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT topic FROM documents_metadata WHERE topic IS NOT NULL")
            rows = cursor.fetchall()
            return [row[0] for row in rows] if rows else ["General"]


sqlite_db = SQLiteDB()
