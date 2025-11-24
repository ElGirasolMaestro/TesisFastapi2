"""
Módulo para cargar y gestionar el modelo BiLSTM entrenado
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

class ModelLoader:
    """Clase para cargar y gestionar el modelo BiLSTM y sus componentes"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Inicializar el cargador de modelo
        
        Args:
            model_dir: Directorio donde se encuentran los archivos del modelo
        """
        self.model_dir = model_dir
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols = ["casos", "precipitacion_mm", "tmin", "tmax"]
        self.window_size = 8
        self.horizon = 1
        self.ubigeo_data: Optional[pd.DataFrame] = None
        
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
            model_path = os.path.join(self.model_dir, "bilstm_model.h5")
            if os.path.exists(model_path):
                logger.info(f"Cargando modelo desde {model_path}")
                
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
                        loss="mse",
                        metrics=["mae"]
                    )
                    logger.info("Modelo BiLSTM cargado exitosamente (sin compilar)")
                except Exception as e1:
                    logger.warning(f"Error cargando con compile=False: {e1}")
                    try:
                        # Método 2: Cargar con safe_mode=False (TF 2.16+)
                        self.model = tf.keras.models.load_model(
                            model_path,
                            safe_mode=False
                        )
                        logger.info("Modelo BiLSTM cargado exitosamente (safe_mode=False)")
                    except Exception as e2:
                        logger.warning(f"Error cargando con safe_mode=False: {e2}")
                        try:
                            # Método 3: Cargar normalmente
                            self.model = tf.keras.models.load_model(model_path)
                            logger.info("Modelo BiLSTM cargado exitosamente (método estándar)")
                        except Exception as e3:
                            logger.warning(f"Error cargando modelo completo: {e3}")
                            # Método 4: Intentar cargar desde pesos
                            try:
                                weights_path = os.path.join(self.model_dir, "bilstm_weights.h5")
                                if os.path.exists(weights_path):
                                    logger.info("Intentando cargar desde pesos...")
                                    # Crear modelo con la arquitectura correcta
                                    self.model = self._create_default_model()
                                    # Cargar pesos
                                    self.model.load_weights(weights_path)
                                    logger.info("Modelo cargado desde pesos exitosamente")
                                else:
                                    raise FileNotFoundError("Archivo de pesos no encontrado")
                            except Exception as e4:
                                logger.error(f"Error cargando desde pesos: {e4}")
                                # Crear modelo por defecto como último recurso
                                logger.warning("Usando modelo por defecto debido a errores de compatibilidad")
                                self.model = self._create_default_model()
            else:
                logger.warning(f"Archivo de modelo no encontrado: {model_path}")
                # Crear modelo por defecto basado en el notebook
                self.model = self._create_default_model()
                logger.info("Modelo por defecto creado")
            
            # Cargar scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                logger.info(f"Cargando scaler desde {scaler_path}")
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler cargado exitosamente")
            else:
                logger.warning(f"Archivo de scaler no encontrado: {scaler_path}")
                # Crear scaler por defecto
                self.scaler = StandardScaler()
                logger.info("Scaler por defecto creado")
            
            # Cargar datos de ubigeos
            await self._load_ubigeo_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            return False
    
    def _create_default_model(self) -> tf.keras.Model:
        """
        Crear modelo por defecto basado en la arquitectura del notebook
        
        Returns:
            tf.keras.Model: Modelo BiLSTM
        """
        tf.random.set_seed(42)
        
        n_features = len(self.feature_cols)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.window_size, n_features)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(self.horizon)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("Modelo por defecto creado con arquitectura BiLSTM")
        return model
    
    async def _load_ubigeo_data(self):
        """Cargar datos de ubigeos para mapear códigos a nombres"""
        try:
            ubigeo_path = os.path.join(self.model_dir, "ubigeos.csv")
            if os.path.exists(ubigeo_path):
                self.ubigeo_data = pd.read_csv(ubigeo_path)
                logger.info("Datos de ubigeos cargados exitosamente")
            else:
                logger.warning("Archivo de ubigeos no encontrado, creando datos por defecto")
                # Crear datos básicos por defecto
                self.ubigeo_data = pd.DataFrame({
                    'ubigeo': ['150101', '150102', '150103'],
                    'departamento': ['LIMA', 'LIMA', 'LIMA'],
                    'provincia': ['LIMA', 'LIMA', 'LIMA'],
                    'distrito': ['LIMA', 'ANCON', 'ATE']
                })
        except Exception as e:
            logger.error(f"Error al cargar datos de ubigeos: {e}")
            self.ubigeo_data = pd.DataFrame()
    
    def get_ubigeo_info(self, ubigeo: str) -> Dict[str, str]:
        """
        Obtener información de un código UBIGEO
        
        Args:
            ubigeo: Código UBIGEO
            
        Returns:
            Dict con información del ubigeo
        """
        if self.ubigeo_data is not None and not self.ubigeo_data.empty:
            match = self.ubigeo_data[self.ubigeo_data['ubigeo'] == ubigeo]
            if not match.empty:
                row = match.iloc[0]
                return {
                    'departamento': row.get('departamento', 'DESCONOCIDO'),
                    'provincia': row.get('provincia', 'DESCONOCIDO'),
                    'distrito': row.get('distrito', 'DESCONOCIDO')
                }
        
        return {
            'departamento': 'DESCONOCIDO',
            'provincia': 'DESCONOCIDO', 
            'distrito': 'DESCONOCIDO'
        }
    
    def save_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Guardar el modelo y scaler
        
        Args:
            model_path: Ruta para guardar el modelo
            scaler_path: Ruta para guardar el scaler
        """
        try:
            if model_path is None:
                model_path = os.path.join(self.model_dir, "bilstm_model.h5")
            
            if scaler_path is None:
                scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            
            if self.model is not None:
                self.model.save(model_path)
                logger.info(f"Modelo guardado en {model_path}")
            
            if self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Scaler guardado en {scaler_path}")
                
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre el modelo cargado
        
        Returns:
            Dict con información del modelo
        """
        return {
            "model_type": "BiLSTM",
            "features": self.feature_cols,
            "window_size": self.window_size,
            "horizon": self.horizon,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "model_summary": str(self.model.summary()) if self.model else None
        }
