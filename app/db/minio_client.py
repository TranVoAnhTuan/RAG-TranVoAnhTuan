from minio import Minio
from app.core.config import settings
import os
from datetime import timedelta

class MinioClient:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False  # Dùng False cho môi trường local (HTTP)
        )
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_path: str, object_name: str) -> str:
        self.client.fput_object(self.bucket_name, object_name, file_path)
        
        # Tạo URL sống tối đa 7 ngày (MinIO không cho phép lâu hơn)
        url = self.client.presigned_get_object(
            self.bucket_name, 
            object_name, 
            expires=timedelta(days=7) # <-- TRUYỀN TIMEDELTA VÀO ĐÂY, KHÔNG ĐỂ SỐ NGUYÊN
        )
        return url

minio_client = MinioClient()