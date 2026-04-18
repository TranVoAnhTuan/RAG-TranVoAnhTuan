import sqlite3
from app.core.config import settings

class SQLiteDB:
    def __init__(self):
        self.db_path = settings.SQLITE_DB_PATH
        self._create_table()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents_metadata (
                    file_hash TEXT PRIMARY KEY,
                    filename TEXT,
                    minio_url TEXT,
                    topic TEXT
                )
            ''')
            try:
                cursor.execute("ALTER TABLE documents_metadata ADD COLUMN topic TEXT DEFAULT 'General'")
            except sqlite3.OperationalError:
                pass 
            conn.commit()

    def get_all_topics(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT topic FROM documents_metadata WHERE topic IS NOT NULL")
            rows = cursor.fetchall()
            return [row[0] for row in rows] if rows else ["General"]

sqlite_db = SQLiteDB()