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
                    minio_url TEXT
                )
            ''')
            conn.commit()

sqlite_db = SQLiteDB()