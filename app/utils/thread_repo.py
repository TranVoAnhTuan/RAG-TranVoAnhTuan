import logging

from app.db.mongo_db import mongo_db

logger = logging.getLogger(__name__)


class ThreadRepository:
    """Data access methods for checking and managing agent conversation threads in MongoDB."""

    @staticmethod
    async def get_all_threads() -> list[dict]:
        """Return a list of recent conversation threads from MongoDB."""
        try:
            db = mongo_db.client["checkpointing_db"]
            pipeline = [
                {"$sort": {"_id": -1}},
                {"$group": {"_id": "$thread_id", "latest_id": {"$first": "$_id"}}},
                {"$sort": {"latest_id": -1}},
                {"$limit": 50},
            ]
            results = list(db["checkpoints"].aggregate(pipeline))
            return [
                {"thread_id": res["_id"], "title": f"Conversation {res['_id'][:8]}"}
                for res in results
                if res["_id"] != "default_thread"
            ]
        except Exception as e:
            logger.error(f"Error fetching threads: {e}")
            return []

    @staticmethod
    async def delete_thread(thread_id: str) -> bool:
        """Delete all checkpoint data for a given thread."""
        try:
            db = mongo_db.client["checkpointing_db"]
            db["checkpoints"].delete_many({"thread_id": thread_id})
            db["checkpoint_writes"].delete_many({"thread_id": thread_id})
            return True
        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")
            return False
