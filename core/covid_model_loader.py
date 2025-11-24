"""
Módulo para cargar y gestionar el modelo BiLSTM de COVID
"""
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class CovidModelLoader:
    """Clase para cargar y gestionar el modelo BiLSTM de COVID y sus componentes"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Inicializar el cargador de modelo COVID
        
        Args:
            model_dir: Directorio donde se encuentran los archivos del modelo
        """
        self.model_dir = model_dir
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols = ["casos", "casos_vecinos", "mes", "dia_semana", "semana_anio", "trimestre"]
        self.window_size = 14  # 14 días de historial
        self.horizon = 1  # Predicción 1 día adelante
        self.neighbor_map: Optional[Dict] = None
        
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
    
    async def load_model(self) -> bool:
        """
        Cargar el modelo y sus componentes
        
        Returns:
            bool: True si la carga fue exitosa
        """
        try:
            # Cargar modelo TensorFlow
            model_path = os.path.join(self.model_dir, "bilstm_covid_provincial.h5")
            if os.path.exists(model_path):
                logger.info(f"Cargando modelo COVID desde {model_path}")
                
                # Intentar cargar con diferentes métodos para compatibilidad
                try:
                    # Método 1: Cargar sin compilar (evita problemas de métricas)
                    self.model = tf.keras.models.load_model(
                        model_path, 
                        compile=False
                    )
                    # Recompilar con las métricas correctas
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss="mse"
                    )
                    logger.info("Modelo COVID BiLSTM cargado exitosamente (sin compilar)")
                except Exception as e1:
                    logger.warning(f"Error cargando con compile=False: {e1}")
                    try:
                        # Método 2: Cargar con safe_mode=False (TF 2.16+)
                        self.model = tf.keras.models.load_model(
                            model_path,
                            safe_mode=False
                        )
                        logger.info("Modelo COVID BiLSTM cargado exitosamente (safe_mode=False)")
                    except Exception as e2:
                        logger.warning(f"Error cargando con safe_mode=False: {e2}")
                        try:
                            # Método 3: Cargar normalmente
                            self.model = tf.keras.models.load_model(model_path)
                            logger.info("Modelo COVID BiLSTM cargado exitosamente (método estándar)")
                        except Exception as e3:
                            logger.error(f"Error cargando modelo con todos los métodos: {e3}")
                            # Crear modelo por defecto como fallback
                            logger.warning("Usando modelo por defecto debido a errores de compatibilidad")
                            self.model = self._create_default_model()
            else:
                logger.warning(f"Archivo de modelo COVID no encontrado: {model_path}")
                # Crear modelo por defecto basado en el notebook
                self.model = self._create_default_model()
                logger.info("Modelo COVID por defecto creado")
            
            # Cargar scaler
            scaler_path = os.path.join(self.model_dir, "scaler_covid_provincial.pkl")
            if os.path.exists(scaler_path):
                logger.info(f"Cargando scaler COVID desde {scaler_path}")
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler COVID cargado exitosamente")
            else:
                logger.warning(f"Archivo de scaler COVID no encontrado: {scaler_path}")
                # Crear scaler por defecto
                self.scaler = StandardScaler()
                logger.info("Scaler COVID por defecto creado")
            
            # Cargar datos de vecinos si están disponibles
            await self._load_neighbor_map()
            
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo COVID: {e}")
            return False
    
    def _create_default_model(self) -> tf.keras.Model:
        """
        Crear modelo por defecto basado en la arquitectura del notebook COVID
        
        Returns:
            tf.keras.Model: Modelo BiLSTM
        """
        tf.random.set_seed(42)
        
        n_features = len(self.feature_cols)
        
        inputs = tf.keras.Input(shape=(self.window_size, n_features))
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)  # predice log(casos_dia+1)
        
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse"
        )
        
        logger.info("Modelo COVID por defecto creado con arquitectura BiLSTM")
        return model
    
    async def _load_neighbor_map(self):
        """Cargar mapa de vecinos si está disponible"""
        try:
            neighbor_path = os.path.join(self.model_dir, "neighbor_map.pkl")
            if os.path.exists(neighbor_path):
                with open(neighbor_path, 'rb') as f:
                    self.neighbor_map = pickle.load(f)
                logger.info("Mapa de vecinos cargado exitosamente")
            else:
                logger.warning("Archivo de mapa de vecinos no encontrado")
                self.neighbor_map = {}
        except Exception as e:
            logger.error(f"Error al cargar mapa de vecinos: {e}")
            self.neighbor_map = {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre el modelo cargado
        
        Returns:
            Dict con información del modelo
        """
        return {
            "model_type": "BiLSTM COVID",
            "features": self.feature_cols,
            "window_size": self.window_size,
            "horizon": self.horizon,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "neighbor_map_loaded": self.neighbor_map is not None and len(self.neighbor_map) > 0
        }

