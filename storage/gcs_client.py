# storage/gcs_client.py
import os
import logging
from typing import Optional
from datetime import timedelta

from google.cloud import storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)


class GCSClient:
    """
    Cliente sencillo para trabajar con Google Cloud Storage.
    - Sube / baja archivos.
    - Genera URLs firmadas para subida directa desde el frontend.
    """

    def __init__(self, bucket_name: Optional[str] = None):
        bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME no está configurado")

        self.client = storage.Client()  # Cloud Run usa la service account por defecto
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

        logger.info(f"GCSClient inicializado con bucket: {bucket_name}")

    # -------------------------- Helpers internos -------------------------- #
    def _normalize_object_name(self, object_name: str) -> str:
        """
        Acepta:
            - "datasets/malaria/archivo.csv"
            - "gs://mi-bucket/datasets/malaria/archivo.csv"
        y devuelve solo "datasets/malaria/archivo.csv"
        """
        if object_name.startswith("gs://"):
            parts = object_name.replace("gs://", "").split("/", 1)
            if len(parts) == 2:
                # ignoramos el nombre de bucket que venga en la URI
                return parts[1]
            return ""
        return object_name

    # --------------------------- Operaciones I/O -------------------------- #
    def upload_file(self, local_path: str, object_name: str) -> str:
        """Subir archivo local al bucket. Devuelve URI gs://..."""
        object_name = self._normalize_object_name(object_name)
        blob = self.bucket.blob(object_name)

        logger.info(f"Subiendo {local_path} a gs://{self.bucket_name}/{object_name}")
        blob.upload_from_filename(local_path)

        return f"gs://{self.bucket_name}/{object_name}"

    def download_file(self, object_name: str, local_path: str) -> None:
        """Descargar objeto del bucket a un archivo local."""
        object_name = self._normalize_object_name(object_name)
        blob = self.bucket.blob(object_name)

        logger.info(f"Descargando gs://{self.bucket_name}/{object_name} a {local_path}")
        blob.download_to_filename(local_path)

    def blob_exists(self, object_name: str) -> bool:
        """Verificar si el objeto existe en el bucket."""
        object_name = self._normalize_object_name(object_name)
        blob = self.bucket.blob(object_name)
        return blob.exists()

    # ------------------------ URL firmada para uploads -------------------- #
    def generate_signed_upload_url(
        self,
        object_name: str,
        content_type: str = "text/csv",
        expiration_minutes: int = 60,
    ) -> str:
        """
        Generar una URL firmada para subir un archivo (PUT) desde el frontend.
        """
        from google.cloud.storage import Blob

        object_name = self._normalize_object_name(object_name)
        blob = Blob(object_name, self.bucket)

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="PUT",
            content_type=content_type,
        )

        logger.info(
            f"URL firmada generada para gs://{self.bucket_name}/{object_name} "
            f"(expira en {expiration_minutes} min)"
        )
        return url
