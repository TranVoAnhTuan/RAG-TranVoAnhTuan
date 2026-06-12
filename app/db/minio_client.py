"""MinIO object storage client for PDF uploads.

Singleton pattern: use the module-level ``minio_client`` instance.
"""

import logging
from datetime import timedelta

from app.core.config import settings
from minio import Minio

logger = logging.getLogger(__name__)


class MinioClient:
    """Handles file uploads to a MinIO bucket and generates presigned URLs."""

    def __init__(self) -> None:
        region = "us-east-1"
        # Client for actual API calls — uses the Docker-internal hostname
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,  # HTTP for local development
            region=region,
        )
        # Client for generating presigned URLs — uses the browser-accessible
        # hostname so the cryptographic signature matches the public URL.
        # NOTE: region must match the internal client so presigned URL
        # generation is purely local (avoids a network call to discover the
        # bucket location, which would fail since MinIO isn't listening on
        # localhost inside the container).
        self.public_client = Minio(
            settings.MINIO_PUBLIC_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
            region=region,
        )
        self.bucket_name: str = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def _ensure_bucket_exists(self) -> None:
        """Ensure the bucket exists, creating it on-the-fly if missing.

        Unlike ``_ensure_bucket`` (called once at init), this guard is used
        before every upload so the system recovers automatically if the
        MinIO data volume was reset or the bucket was deleted externally.
        """
        if not self.client.bucket_exists(self.bucket_name):
            logger.info("🪣 Bucket '%s' not found — creating it.", self.bucket_name)
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_path: str, object_name: str) -> str:
        """Upload a file and return a presigned URL valid for 7 days."""
        self._ensure_bucket_exists()
        self.client.fput_object(self.bucket_name, object_name, file_path)

        # Generate the presigned URL against the public hostname so the
        # cryptographic signature covers localhost:9000 — no host-rewrite
        # after the fact, which would break the signature.
        return self.public_client.presigned_get_object(
            self.bucket_name,
            object_name,
            expires=timedelta(days=7),
        )


minio_client = MinioClient()
