"""MinIO object storage client for PDF uploads.

Singleton pattern: use the module-level ``minio_client`` instance.
"""

from datetime import timedelta

from app.core.config import settings
from minio import Minio


class MinioClient:
    """Handles file uploads to a MinIO bucket and generates presigned URLs."""

    def __init__(self) -> None:
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,  # HTTP for local development
        )
        self.bucket_name: str = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_path: str, object_name: str) -> str:
        """Upload a file and return a presigned URL valid for 7 days."""
        self.client.fput_object(self.bucket_name, object_name, file_path)

        # MinIO caps presigned URL expiry at 7 days
        url = self.client.presigned_get_object(
            self.bucket_name,
            object_name,
            expires=timedelta(days=7),
        )
        return url


minio_client = MinioClient()
