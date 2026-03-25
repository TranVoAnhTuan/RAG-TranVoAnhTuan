import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_KEY = os.getenv("API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

    SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME")
    DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")

settings = Settings()