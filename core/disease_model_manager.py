"""
Sistema de gestión de modelos por enfermedad
Permite cargar y usar múltiples modelos dinámicamente
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

class DiseaseModelManager:
    """Gestor de modelos para múltiples enfermedades"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializar gestor de modelos
        
        Args:
            models_dir: Directorio donde se encuentran los modelos
        """
        self.models_dir = models_dir
        self.models: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        # Modelos base conocidos
        self.base_models = {
            "dengue": {
                "model_file": "bilstm_model.h5",
                "scaler_file": "scaler.pkl",
                "config_file": None,
                "window_size": 8,
                "horizon": 1,
                "features": ["casos", "precipitacion_mm", "tmin", "tmax"],
                "frequency": "weekly"
            },
            "covid": {
                "model_file": "bilstm_covid_provincial.h5",
                "scaler_file": "scaler_covid_provincial.pkl",
                "config_file": None,
                "window_size": 14,
                "horizon": 1,
                "features": ["casos", "casos_vecinos", "mes", "dia_semana", "semana_anio", "trimestre"],
                "frequency": "daily"
            }
        }
    
    def discover_models(self) -> List[str]:
        """
        Descubrir modelos disponibles en el directorio
        
        Returns:
            Lista de nombres de enfermedades disponibles
        """
        available_diseases = []
        
        # Buscar modelos base
        for disease, config in self.base_models.items():
            model_path = os.path.join(self.models_dir, config["model_file"])
            if os.path.exists(model_path):
                available_diseases.append(disease)
        
        # Buscar modelos entrenados con transfer learning
        for file in os.listdir(self.models_dir):
            if file.startswith("bilstm_") and file.endswith(".h5"):
                disease_name = file.replace("bilstm_", "").replace(".h5", "")
                if disease_name not in available_diseases:
                    # Verificar si tiene config
                    config_file = os.path.join(self.models_dir, f"config_{disease_name}.json")
                    if os.path.exists(config_file):
                        available_diseases.append(disease_name)
        
        logger.info(f"Modelos descubiertos: {available_diseases}")
        return available_diseases
    
    async def load_model(self, disease_name: str) -> bool:
        """
        Cargar modelo para una enfermedad específica
        
        Args:
            disease_name: Nombre de la enfermedad
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            # Verificar si ya está cargado
            if disease_name in self.models:
                logger.info(f"Modelo de {disease_name} ya está cargado")
                return True
            
            # Buscar configuración
            config = None
            
            # Intentar cargar desde config file
            config_file = os.path.join(self.models_dir, f"config_{disease_name}.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuración cargada desde {config_file}")
            elif disease_name in self.base_models:
                config = self.base_models[disease_name]
            
            if not config:
                logger.error(f"No se encontró configuración para {disease_name}")
                return False
            
            # Cargar modelo
            model_file = config.get("model_file") or f"bilstm_{disease_name}.h5"
            model_path = os.path.join(self.models_dir, model_file)
            
            if not os.path.exists(model_path):
                logger.error(f"Modelo no encontrado: {model_path}")
                return False
            
            logger.info(f"Cargando modelo de {disease_name} desde {model_path}")
            
            # Cargar modelo TensorFlow
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse",
                    metrics=["mae"]
                )
            except Exception as e:
                logger.warning(f"Error cargando modelo: {e}, intentando sin compilar...")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse"
                )
            
            # Cargar scaler
            scaler_file = config.get("scaler_file") or f"scaler_{disease_name}.pkl"
            scaler_path = os.path.join(self.models_dir, scaler_file)
            
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                logger.warning(f"Scaler no encontrado, usando por defecto")
            
            # Guardar en cache
            self.models[disease_name] = {
                "model": model,
                "scaler": scaler,
                "config": config
            }
            self.configs[disease_name] = config
            
            logger.info(f"Modelo de {disease_name} cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo de {disease_name}: {e}")
            return False
    
    def get_model(self, disease_name: str) -> Optional[tf.keras.Model]:
        """Obtener modelo cargado"""
        if disease_name in self.models:
            return self.models[disease_name]["model"]
        return None
    
    def get_scaler(self, disease_name: str) -> Optional[Any]:
        """Obtener scaler cargado"""
        if disease_name in self.models:
            return self.models[disease_name]["scaler"]
        return None
    
    def get_config(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """Obtener configuración del modelo"""
        return self.configs.get(disease_name)
    
    def is_loaded(self, disease_name: str) -> bool:
        """Verificar si un modelo está cargado"""
        return disease_name in self.models
    
    def get_model_info(self, disease_name: str) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if disease_name not in self.models:
            return {"error": "Modelo no cargado"}
        
        config = self.configs.get(disease_name, {})
        model = self.models[disease_name]["model"]
        
        return {
            "disease_name": disease_name,
            "model_type": "BiLSTM",
            "window_size": config.get("window_size", "unknown"),
            "horizon": config.get("horizon", "unknown"),
            "features": config.get("features", []),
            "frequency": config.get("frequency", "unknown"),
            "model_loaded": True,
            "scaler_loaded": self.models[disease_name]["scaler"] is not None,
            "input_shape": str(model.input_shape) if model else None,
            "output_shape": str(model.output_shape) if model else None
        }
    
    async def load_all_available(self):
        """Cargar todos los modelos disponibles"""
        diseases = self.discover_models()
        for disease in diseases:
            await self.load_model(disease)

