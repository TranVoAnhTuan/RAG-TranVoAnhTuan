import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_KEY = os.getenv("API_KEY")

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "rag_database.db")
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag-agent")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "89898989")
    MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "pdfs")
    MONGODB_HOST = os.getenv("DATABASE_HOST")
    
    HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

    SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME")
    DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")

settings = Settings()