from pymongo import MongoClient
import gridfs
from app.core.config import settings

class MongoDB:
    def __init__(self):
        # Update your MongoDB URI here
        self.client = MongoClient(settings.MONGODB_HOST) 
        self.db = self.client["agentic_rag_database"]
        self.fs = gridfs.GridFS(self.db)
        self.collection = self.db["documents_metadata"]

mongo_db = MongoDB()