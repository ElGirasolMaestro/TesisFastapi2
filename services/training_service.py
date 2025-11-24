"""
Servicio para entrenar modelos automáticamente desde CSV subido
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import joblib
import json
import asyncio
from typing import Dict, Any, Optional
import logging

from core.transfer_learning import TransferLearningTrainer, DiseaseDataPreprocessor

logger = logging.getLogger(__name__)

# Variable global para el servicio genérico (se establecerá desde main.py)
_generic_service = None

def set_generic_service(service):
    """Establecer el servicio genérico para recargar modelos después del entrenamiento"""
    global _generic_service
    _generic_service = service

class TrainingService:
    """Servicio para entrenar modelos automáticamente"""
    
    def __init__(self, models_dir: str = "models", base_model_path: Optional[str] = None):
        """
        Inicializar servicio de entrenamiento
        
        Args:
            models_dir: Directorio para guardar modelos
            base_model_path: Ruta al modelo base para transfer learning (None = auto-detectar)
        """
        self.models_dir = models_dir
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        os.makedirs(models_dir, exist_ok=True)
        
        # Auto-detectar mejor modelo base disponible
        self.base_model_path = base_model_path or self._find_best_base_model()
    
    def _find_best_base_model(self) -> str:
        """
        Encontrar el mejor modelo base disponible para transfer learning
        
        Prioridad: 1. dengue (bilstm_model.h5), 2. covid (bilstm_covid_provincial.h5)
        
        Returns:
            Ruta al modelo base encontrado
        """
        # Prioridad: 1. dengue, 2. covid
        models_to_check = [
            os.path.join(self.models_dir, "bilstm_model.h5"),  # Dengue (preferido)
            os.path.join(self.models_dir, "bilstm_covid_provincial.h5")  # COVID (alternativa)
        ]
        
        for model_path in models_to_check:
            abs_path = os.path.abspath(model_path)
            if os.path.exists(abs_path):
                logger.info(f"✅ Modelo base encontrado para transfer learning: {abs_path}")
                return abs_path
        
        # Si no encuentra ninguno, retornar ruta por defecto (se creará modelo por defecto)
        default_path = os.path.join(self.models_dir, "bilstm_model.h5")
        logger.warning(
            f"⚠️ No se encontró modelo base pre-entrenado. "
            f"Se usará: {default_path} (se creará modelo por defecto si es necesario)"
        )
        return default_path
    
    async def train_from_csv(
        self,
        csv_path: str,
        disease_name: str,
        window_size: int = 8,
        horizon: int = 1,
        test_split: float = 0.2,
        epochs: int = 20,
        batch_size: int = 64,
        base_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Entrenar modelo desde CSV automáticamente
        
        Args:
            csv_path: Ruta al CSV subido
            disease_name: Nombre de la enfermedad
            window_size: Tamaño de ventana temporal
            horizon: Horizonte de predicción
            test_split: Proporción para test (0.2 = 20%)
            epochs: Número de épocas
            batch_size: Tamaño de batch
            base_model_path: Modelo base (opcional, usa default si None)
            
        Returns:
            Dict con información del entrenamiento
        """
        job_id = f"{disease_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Inicializar estado del job
        self.training_jobs[job_id] = {
            "status": "starting",
            "disease_name": disease_name,
            "progress": 0,
            "message": "Iniciando entrenamiento...",
            "started_at": datetime.now().isoformat()
        }
        
        # Ejecutar entrenamiento en background (no bloquear)
        asyncio.create_task(self._train_model_async(
            job_id, csv_path, disease_name, window_size, horizon,
            test_split, epochs, batch_size, base_model_path or self.base_model_path
        ))
        
        # Retornar inmediatamente con job_id
        return {
            "job_id": job_id,
            "status": "starting",
            "message": "Entrenamiento iniciado en background"
        }
    
    async def _train_model_async(
        self,
        job_id: str,
        csv_path: str,
        disease_name: str,
        window_size: int,
        horizon: int,
        test_split: float,
        epochs: int,
        batch_size: int,
        base_model_path: str
    ):
        """Ejecutar entrenamiento de forma asíncrona en background"""
        try:
            self.training_jobs[job_id]["status"] = "running"
            
            # Convertir a ruta absoluta y verificar que existe
            csv_path = os.path.abspath(csv_path)
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")
            
            # Actualizar progreso
            self._update_progress(job_id, 10, f"Cargando y preprocesando datos desde {os.path.basename(csv_path)}...")
            
            # 1. Preprocesar datos
            preprocessor = DiseaseDataPreprocessor(
                window_size=window_size,
                horizon=horizon
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
            
            # Determinar qué modelo base usar (priorizar el especificado, luego auto-detectar)
            if base_model_path is None:
                base_model_path = self.base_model_path  # Usar el detectado en __init__
            
            # Verificar que el modelo base existe
            base_model_abs_path = os.path.abspath(base_model_path)
            if not os.path.exists(base_model_abs_path):
                # Intentar con rutas alternativas (dengue primero, luego COVID)
                alt_paths = [
                    os.path.abspath("models/bilstm_model.h5"),  # Dengue (preferido)
                    os.path.abspath("models/bilstm_covid_provincial.h5")  # COVID (alternativa)
                ]
                
                found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        logger.info(f"✅ Modelo base {base_model_abs_path} no encontrado, usando {alt_path}")
                        base_model_path = alt_path
                        found = True
                        break
                
                if not found:
                    logger.warning(
                        f"⚠️ No se encontró ningún modelo base pre-entrenado. "
                        f"Se creará un modelo base por defecto para transfer learning."
                    )
                    base_model_path = None  # None para que se cree modelo por defecto
            else:
                logger.info(f"✅ Usando modelo base para transfer learning: {base_model_abs_path}")
            
            trainer = TransferLearningTrainer(base_model_path=base_model_path)
            transfer_model = trainer.create_transfer_model(
                input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                freeze_encoder=True
            )
            
            # 6. Entrenar
            self._update_progress(job_id, 70, "Iniciando entrenamiento...")
            
            # Ejecutar entrenamiento en thread separado para no bloquear
            loop = asyncio.get_event_loop()
            trained_model = await loop.run_in_executor(
                None,
                lambda: trainer.train_transfer_model(
                    X_train_scaled, y_train,
                    X_test_scaled, y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=5
                )
            )
            
            self._update_progress(job_id, 90, "Evaluando modelo...")
            
            # 7. Evaluar
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            y_pred = trained_model.predict(X_test_scaled, verbose=0).ravel()
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # 8. Guardar
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
                "base_model": base_model_path,
                "use_log": False,
                "frequency": "weekly" if window_size >= 7 else "daily",
                "metrics": {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2)
                },
                "trained_date": datetime.now().isoformat(),
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "job_id": job_id
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Intentar cargar el modelo en el servicio genérico si está disponible
            if _generic_service is not None:
                try:
                    logger.info(f"Intentando cargar modelo de {disease_name} en servicio genérico...")
                    await _generic_service.model_manager.load_model(disease_name)
                    logger.info(f"✅ Modelo de {disease_name} cargado exitosamente en servicio genérico")
                except Exception as e:
                    logger.warning(f"No se pudo cargar modelo de {disease_name} en servicio genérico: {e}")
            
            # Limpiar archivo temporal
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            except:
                pass
            
            # Actualizar estado final
            self.training_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Entrenamiento completado exitosamente",
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "model_path": model_path,
                    "scaler_path": scaler_path,
                    "config_path": config_path,
                    "metrics": {
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "r2": float(r2)
                    },
                    "train_samples": int(X_train.shape[0]),
                    "test_samples": int(X_test.shape[0])
                }
            })
            
        except Exception as e:
            logger.error(f"Error en entrenamiento {job_id}: {e}")
            self.training_jobs[job_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Error: {str(e)}",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })
            
            # Limpiar archivo en caso de error
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            except:
                pass
    
    def _update_progress(self, job_id: str, progress: int, message: str):
        """Actualizar progreso del job"""
        if job_id in self.training_jobs:
            self.training_jobs[job_id]["progress"] = progress
            self.training_jobs[job_id]["message"] = message
            logger.info(f"Job {job_id}: {progress}% - {message}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un job de entrenamiento"""
        return self.training_jobs.get(job_id)
    
    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Listar todos los jobs de entrenamiento"""
        return self.training_jobs
