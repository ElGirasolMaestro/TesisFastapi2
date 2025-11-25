# services/training_service.py
"""
Servicio para entrenar modelos automáticamente desde CSV (local o GCS)
"""
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.transfer_learning import TransferLearningTrainer, DiseaseDataPreprocessor
from storage.gcs_client import GCSClient

logger = logging.getLogger(__name__)

# Variable global para el servicio genérico (se establecerá desde main.py)
_generic_service = None


def set_generic_service(service):
    """Establecer el servicio genérico para recargar modelos después del entrenamiento"""
    global _generic_service
    _generic_service = service


class TrainingService:
    """Servicio para entrenar modelos automáticamente"""

    def __init__(
        self,
        models_dir: str = "models",
        base_model_path: Optional[str] = None,
        gcs_client: Optional[GCSClient] = None,
    ):
        """
        Inicializar servicio de entrenamiento

        Args:
            models_dir: Directorio local para guardar modelos
            base_model_path: Ruta al modelo base para transfer learning (None = auto-detectar)
            gcs_client: Cliente de Google Cloud Storage (opcional)
        """
        self.models_dir = models_dir
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        self.gcs = gcs_client

        os.makedirs(models_dir, exist_ok=True)

        # Auto-detectar mejor modelo base disponible
        self.base_model_path = base_model_path or self._find_best_base_model()

    # ------------------------------------------------------------------ #
    # Discovery de modelo base
    # ------------------------------------------------------------------ #
    def _find_best_base_model(self) -> str:
        """
        Encontrar el mejor modelo base disponible para transfer learning

        Prioridad: 1. dengue (bilstm_model.h5), 2. covid (bilstm_covid_provincial.h5)
        """
        models_to_check = [
            os.path.join(self.models_dir, "bilstm_model.h5"),  # Dengue (preferido)
            os.path.join(self.models_dir, "bilstm_covid_provincial.h5"),  # COVID
        ]

        for model_path in models_to_check:
            abs_path = os.path.abspath(model_path)
            if os.path.exists(abs_path):
                logger.info(f"✅ Modelo base encontrado para transfer learning: {abs_path}")
                return abs_path

        default_path = os.path.join(self.models_dir, "bilstm_model.h5")
        logger.warning(
            "⚠️ No se encontró modelo base pre-entrenado. "
            f"Se usará: {default_path} (se creará si es necesario)."
        )
        return default_path

    # ------------------------------------------------------------------ #
    # Entrenamiento desde CSV subido vía API (para archivos pequeños)
    # ------------------------------------------------------------------ #
    async def train_from_csv(
        self,
        csv_path: str,
        disease_name: str,
        window_size: int = 8,
        horizon: int = 1,
        test_split: float = 0.2,
        epochs: int = 20,
        batch_size: int = 64,
        base_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Entrenar modelo desde un CSV ya guardado en disco (Cloud Run / local)."""
        job_id = f"{disease_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_job(job_id, disease_name)

        asyncio.create_task(
            self._train_model_async(
                job_id=job_id,
                local_csv_path=os.path.abspath(csv_path),
                gcs_uri=None,
                disease_name=disease_name,
                window_size=window_size,
                horizon=horizon,
                test_split=test_split,
                epochs=epochs,
                batch_size=batch_size,
                base_model_path=base_model_path or self.base_model_path,
            )
        )

        return {
            "job_id": job_id,
            "status": "starting",
            "message": "Entrenamiento iniciado en background (CSV local)",
        }

    # ------------------------------------------------------------------ #
    # Entrenamiento desde archivo en GCS (versión PRO)
    # ------------------------------------------------------------------ #
    async def train_from_gcs(
        self,
        gcs_uri: str,
        disease_name: str,
        window_size: int = 8,
        horizon: int = 1,
        test_split: float = 0.2,
        epochs: int = 20,
        batch_size: int = 64,
        base_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Entrenar modelo usando un CSV almacenado en Cloud Storage.

        gcs_uri puede ser:
            - "gs://bucket/datasets/malaria/mi_archivo.csv"
            - "datasets/malaria/mi_archivo.csv"
        """
        if self.gcs is None:
            raise RuntimeError("GCSClient no configurado en TrainingService")

        job_id = f"{disease_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_job(job_id, disease_name)

        # CSV se descargará a /tmp para el entrenamiento
        local_csv_path = os.path.join("/tmp", f"{job_id}.csv")

        asyncio.create_task(
            self._train_model_async(
                job_id=job_id,
                local_csv_path=local_csv_path,
                gcs_uri=gcs_uri,
                disease_name=disease_name,
                window_size=window_size,
                horizon=horizon,
                test_split=test_split,
                epochs=epochs,
                batch_size=batch_size,
                base_model_path=base_model_path or self.base_model_path,
            )
        )

        return {
            "job_id": job_id,
            "status": "starting",
            "message": "Entrenamiento iniciado en background (CSV en GCS)",
        }

    # ------------------------------------------------------------------ #
    # Lógica interna de entrenamiento
    # ------------------------------------------------------------------ #
    def _init_job(self, job_id: str, disease_name: str) -> None:
        self.training_jobs[job_id] = {
            "status": "starting",
            "disease_name": disease_name,
            "progress": 0,
            "message": "Iniciando entrenamiento...",
            "started_at": datetime.now().isoformat(),
        }

    async def _train_model_async(
        self,
        job_id: str,
        local_csv_path: str,
        gcs_uri: Optional[str],
        disease_name: str,
        window_size: int,
        horizon: int,
        test_split: float,
        epochs: int,
        batch_size: int,
        base_model_path: str,
    ) -> None:
        """Ejecutar entrenamiento de forma asíncrona en background."""
        try:
            self.training_jobs[job_id]["status"] = "running"

            # Si viene de GCS, descargar primero
            if gcs_uri and self.gcs is not None:
                self._update_progress(
                    job_id, 5, f"Descargando CSV desde GCS: {gcs_uri}"
                )
                self.gcs.download_file(gcs_uri, local_csv_path)
                csv_path = local_csv_path
            else:
                csv_path = local_csv_path

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")

            self._update_progress(
                job_id,
                10,
                f"Cargando y preprocesando datos desde {os.path.basename(csv_path)}...",
            )

            # 1. Preprocesar datos
            preprocessor = DiseaseDataPreprocessor(
                window_size=window_size,
                horizon=horizon,
            )

            df = preprocessor.load_csv(csv_path)
            self._update_progress(job_id, 20, "Datos cargados, creando secuencias...")

            # 2. Crear secuencias
            X, y, dates, keys = preprocessor.create_sequences(df)
            self._update_progress(job_id, 30, f"Secuencias creadas: {X.shape[0]} muestras")

            # 3. Split train/test
            split_idx = int((1 - test_split) * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            self._update_progress(job_id, 40, "Datos divididos en train/test")

            # 4. Escalar
            preprocessor.fit_scaler(X_train)
            X_train_scaled = preprocessor.transform_sequences(X_train)
            X_test_scaled = preprocessor.transform_sequences(X_test)
            self._update_progress(job_id, 50, "Datos escalados")

            # 5. Crear modelo de transfer learning
            self._update_progress(job_id, 60, "Creando modelo de transfer learning...")

            base_model_abs_path = os.path.abspath(base_model_path)
            if not os.path.exists(base_model_abs_path):
                # Buscar alternativas (dengue / covid)
                alt_paths = [
                    os.path.abspath("models/bilstm_model.h5"),
                    os.path.abspath("models/bilstm_covid_provincial.h5"),
                ]
                found = False
                for alt in alt_paths:
                    if os.path.exists(alt):
                        logger.info(
                            f"✅ Modelo base {base_model_abs_path} no encontrado, usando {alt}"
                        )
                        base_model_abs_path = alt
                        found = True
                        break
                if not found:
                    logger.warning(
                        "⚠️ No se encontró modelo base pre-entrenado. "
                        "Se creará un modelo base por defecto."
                    )
                    base_model_abs_path = None
            else:
                logger.info(f"✅ Usando modelo base: {base_model_abs_path}")

            trainer = TransferLearningTrainer(base_model_path=base_model_abs_path)
            transfer_model = trainer.create_transfer_model(
                input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                freeze_encoder=True,
            )

            # 6. Entrenar
            self._update_progress(job_id, 70, "Iniciando entrenamiento...")

            loop = asyncio.get_event_loop()
            trained_model = await loop.run_in_executor(
                None,
                lambda: trainer.train_transfer_model(
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=5,
                ),
            )

            self._update_progress(job_id, 90, "Evaluando modelo...")

            # 7. Evaluar
            y_pred = trained_model.predict(X_test_scaled, verbose=0).ravel()
            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = r2_score(y_test, y_pred)

            # 8. Guardar localmente
            self._update_progress(job_id, 95, "Guardando modelo...")

            model_path = os.path.join(self.models_dir, f"bilstm_{disease_name}.h5")
            scaler_path = os.path.join(self.models_dir, f"scaler_{disease_name}.pkl")
            config_path = os.path.join(self.models_dir, f"config_{disease_name}.json")

            trained_model.save(model_path)
            joblib.dump(preprocessor.scaler, scaler_path)

            config = {
                "disease_name": disease_name,
                "window_size": window_size,
                "horizon": horizon,
                "features": preprocessor.feature_cols,
                "base_model": base_model_abs_path,
                "use_log": False,
                "frequency": "weekly" if window_size >= 7 else "daily",
                "metrics": {
                    "mae": float(mae),
                    "rmse": rmse,
                    "r2": float(r2),
                },
                "trained_date": datetime.now().isoformat(),
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "job_id": job_id,
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # 9. Subir artefactos a GCS (si hay cliente)
            gcs_artifacts = {}
            if self.gcs is not None:
                base_prefix = f"models/{disease_name}"
                gcs_artifacts["model_uri"] = self.gcs.upload_file(
                    model_path, f"{base_prefix}/bilstm_{disease_name}.h5"
                )
                gcs_artifacts["scaler_uri"] = self.gcs.upload_file(
                    scaler_path, f"{base_prefix}/scaler_{disease_name}.pkl"
                )
                gcs_artifacts["config_uri"] = self.gcs.upload_file(
                    config_path, f"{base_prefix}/config_{disease_name}.json"
                )

            # 10. Recargar modelo en el servicio genérico
            if _generic_service is not None:
                try:
                    logger.info(
                        f"Intentando cargar modelo de {disease_name} en servicio genérico..."
                    )
                    await _generic_service.model_manager.load_model(disease_name)
                    logger.info("✅ Modelo cargado en servicio genérico")
                except Exception as e:
                    logger.warning(
                        f"No se pudo cargar modelo de {disease_name} en servicio genérico: {e}"
                    )

            # Limpiar CSV local
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            except Exception:
                pass

            # Estado final
            self.training_jobs[job_id].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "message": "Entrenamiento completado exitosamente",
                    "completed_at": datetime.now().isoformat(),
                    "result": {
                        "model_path": model_path,
                        "scaler_path": scaler_path,
                        "config_path": config_path,
                        "gcs_artifacts": gcs_artifacts,
                        "metrics": {
                            "mae": float(mae),
                            "rmse": rmse,
                            "r2": float(r2),
                        },
                        "train_samples": int(X_train.shape[0]),
                        "test_samples": int(X_test.shape[0]),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error en entrenamiento {job_id}: {e}")
            self.training_jobs[job_id].update(
                {
                    "status": "failed",
                    "progress": 0,
                    "message": f"Error: {str(e)}",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                }
            )
            try:
                if os.path.exists(local_csv_path):
                    os.remove(local_csv_path)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Utilidades de job
    # ------------------------------------------------------------------ #
    def _update_progress(self, job_id: str, progress: int, message: str) -> None:
        if job_id in self.training_jobs:
            self.training_jobs[job_id]["progress"] = progress
            self.training_jobs[job_id]["message"] = message
            logger.info(f"Job {job_id}: {progress}% - {message}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.training_jobs.get(job_id)

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        return self.training_jobs
