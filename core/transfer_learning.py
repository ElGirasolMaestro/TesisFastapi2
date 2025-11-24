"""
Sistema de Transfer Learning para adaptar modelos existentes a nuevas enfermedades
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TransferLearningTrainer:
    """Clase para entrenar modelos usando transfer learning desde modelos base"""
    
    def __init__(
        self,
        base_model_path: Optional[str] = None,
        base_model: Optional[tf.keras.Model] = None,
        freeze_layers: int = 1  # Cuántas capas congelar (1 = solo la última)
    ):
        """
        Inicializar el entrenador de transfer learning
        
        Args:
            base_model_path: Ruta al modelo base (dengue o covid)
            base_model: Modelo base ya cargado
            freeze_layers: Número de capas a congelar (0 = todas entrenables)
        """
        self.base_model_path = base_model_path
        self.base_model = base_model
        self.freeze_layers = freeze_layers
        
    def load_base_model(self, model_path: str) -> tf.keras.Model:
        """
        Cargar modelo base para transfer learning
        
        Args:
            model_path: Ruta al modelo base
            
        Returns:
            Modelo base cargado
        """
        # Convertir a ruta absoluta
        model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            error_msg = f"Modelo base no encontrado: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.info(f"Cargando modelo base desde {model_path}")
            
            # Intentar cargar con diferentes estrategias
            try:
                base_model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e1:
                logger.warning(f"Error cargando con compile=False: {e1}, intentando safe_mode=False...")
                try:
                    base_model = tf.keras.models.load_model(model_path, safe_mode=False)
                except Exception as e2:
                    logger.warning(f"Error cargando con safe_mode=False: {e2}, intentando método estándar...")
                    base_model = tf.keras.models.load_model(model_path)
            
            logger.info("Modelo base cargado exitosamente")
            return base_model
        except Exception as e:
            error_msg = f"Error cargando modelo base desde {model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _create_default_base_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Crear modelo base por defecto cuando no hay modelo pre-entrenado disponible
        
        Args:
            input_shape: Shape de entrada (window_size, n_features)
            
        Returns:
            Modelo base por defecto
        """
        window_size, n_features = input_shape
        
        logger.info(f"Creando modelo base por defecto con arquitectura BiLSTM (input: {input_shape})")
        
        # Arquitectura similar a los modelos existentes
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window_size, n_features)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("Modelo base por defecto creado exitosamente")
        return model
    
    def create_transfer_model(
        self,
        input_shape: Tuple[int, int],
        base_model: Optional[tf.keras.Model] = None,
        freeze_encoder: bool = True
    ) -> tf.keras.Model:
        """
        Crear modelo de transfer learning basado en modelo pre-entrenado
        
        Args:
            input_shape: Shape de entrada (window_size, n_features)
            base_model: Modelo base (si None, usa self.base_model)
            freeze_encoder: Si True, congela las capas del encoder
            
        Returns:
            Modelo de transfer learning
        """
        if base_model is None:
            if self.base_model is not None:
                base_model = self.base_model
            elif self.base_model_path:
                # Verificar que la ruta existe antes de intentar cargar
                model_path = os.path.abspath(self.base_model_path)
                if os.path.exists(model_path):
                    try:
                        base_model = self.load_base_model(self.base_model_path)
                        # Guardar en self.base_model para futuras referencias
                        self.base_model = base_model
                    except Exception as e:
                        logger.warning(f"No se pudo cargar modelo base desde {model_path}: {e}")
                        logger.info("Creando modelo base por defecto...")
                        base_model = self._create_default_base_model(input_shape)
                        self.base_model = base_model
                else:
                    logger.warning(f"Modelo base no encontrado: {model_path}")
                    logger.info("Creando modelo base por defecto...")
                    base_model = self._create_default_base_model(input_shape)
                    self.base_model = base_model
            else:
                logger.warning("No se proporcionó modelo base, creando modelo por defecto...")
                base_model = self._create_default_base_model(input_shape)
                self.base_model = base_model
        
        # Obtener el input shape esperado por el modelo base
        base_input_shape = base_model.input_shape[1:]  # (window_size, n_features_base)
        base_n_features = base_input_shape[1] if len(base_input_shape) > 1 else base_input_shape[0]
        current_n_features = input_shape[1]
        
        # Obtener las capas del encoder (excluir Input y última capa Dense)
        # Filtrar capas Input y tomar todas excepto la última Dense
        encoder_layers = []
        for layer in base_model.layers:
            # Excluir capas Input y la última capa Dense (output)
            if not isinstance(layer, tf.keras.layers.InputLayer):
                # Si es la última capa y es Dense, no la incluimos
                if layer == base_model.layers[-1] and isinstance(layer, tf.keras.layers.Dense):
                    continue
                encoder_layers.append(layer)
        
        # Crear nuevo modelo con encoder pre-entrenado
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Si las features no coinciden, agregar capa de adaptación
        if current_n_features != base_n_features:
            logger.info(
                f"🔧 Adaptando features: {current_n_features} -> {base_n_features} "
                f"(modelo base espera {base_n_features} features)"
            )
            # Capa de adaptación: transforma las features actuales a las esperadas por el modelo base
            # Usamos una capa Dense que se aplica a cada timestep
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(base_n_features, activation="linear", name="feature_adaptation")
            )(x)
        
        # Aplicar capas del encoder
        for i, layer in enumerate(encoder_layers):
            if freeze_encoder and i < len(encoder_layers) - self.freeze_layers:
                layer.trainable = False  # Congelar capas tempranas
            x = layer(x)
        
        # Nueva capa de salida (entrenable)
        outputs = tf.keras.layers.Dense(1, name='disease_output')(x)
        
        transfer_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar con learning rate más bajo para fine-tuning
        transfer_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # LR más bajo
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info(f"Modelo de transfer learning creado. Capas congeladas: {len(encoder_layers) - self.freeze_layers}")
        return transfer_model
    
    def train_transfer_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 5
    ) -> tf.keras.Model:
        """
        Entrenar modelo de transfer learning
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Targets de entrenamiento
            X_val: Datos de validación (opcional)
            y_val: Targets de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño de batch
            patience: Paciencia para early stopping
            
        Returns:
            Modelo entrenado
        """
        # Crear modelo de transfer learning (esto cargará el modelo base si es necesario)
        transfer_model = self.create_transfer_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        
        # Callbacks
        callbacks = []
        if X_val is not None and y_val is not None:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1  # Mostrar mensaje cuando se detiene
            )
            callbacks.append(early_stop)
        
        # Entrenar
        logger.info(f"Iniciando entrenamiento con transfer learning... (máximo {epochs} épocas, patience={patience})")
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = transfer_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Informar sobre el entrenamiento
        epochs_trained = len(history.history['loss'])
        if epochs_trained < epochs:
            logger.info(f"✅ Entrenamiento detenido por EarlyStopping en época {epochs_trained}/{epochs}")
            logger.info(f"   El modelo no mejoró en validación durante {patience} épocas consecutivas")
            logger.info(f"   Se restauraron los mejores pesos (época con menor val_loss)")
        else:
            logger.info(f"✅ Entrenamiento completado: {epochs_trained}/{epochs} épocas")
        
        return transfer_model


class DiseaseDataPreprocessor:
    """Preprocesador genérico para datos de diferentes enfermedades"""
    
    def __init__(
        self,
        window_size: int = 8,
        horizon: int = 1,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Inicializar preprocesador
        
        Args:
            window_size: Tamaño de ventana temporal
            horizon: Horizonte de predicción
            feature_cols: Columnas a usar como features
        """
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols = feature_cols or []
        self.scaler = StandardScaler()
        
    def load_csv(
        self,
        csv_path: str,
        date_col: str = "fecha",
        location_cols: List[str] = None,
        case_col: str = "casos",
        date_format: str = None
    ) -> pd.DataFrame:
        """
        Cargar CSV de enfermedad y normalizar estructura
        
        Args:
            csv_path: Ruta al CSV
            date_col: Nombre de la columna de fecha (o se infiere)
            location_cols: Columnas de ubicación [departamento, provincia, distrito]
            case_col: Columna con casos (o se infiere)
            date_format: Formato de fecha si no se infiere automáticamente
            
        Returns:
            DataFrame normalizado
        """
        logger.info(f"Cargando CSV desde {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Detectar columnas automáticamente si no se especifican
        if location_cols is None:
            location_cols = self._detect_location_cols(df)
        
        if case_col is None:
            case_col = self._detect_case_col(df)
        
        # Normalizar estructura
        df_normalized = self._normalize_dataframe(
            df, date_col, location_cols, case_col, date_format
        )
        
        logger.info(f"CSV cargado: {len(df_normalized)} registros")
        return df_normalized
    
    def _detect_location_cols(self, df: pd.DataFrame) -> List[str]:
        """Detectar columnas de ubicación automáticamente"""
        possible_cols = ["departamento", "provincia", "distrito"]
        found_cols = [col for col in possible_cols if col in df.columns]
        return found_cols if found_cols else []
    
    def _detect_case_col(self, df: pd.DataFrame) -> str:
        """Detectar columna de casos automáticamente"""
        # Buscar columnas que puedan contener casos
        possible_cols = ["casos", "caso", "count", "total", "diagnostic"]
        for col in possible_cols:
            if col in df.columns:
                return col
        # Si no se encuentra, asumir que hay que agregar por ubicación
        return None
    
    def _normalize_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str,
        location_cols: List[str],
        case_col: str,
        date_format: str
    ) -> pd.DataFrame:
        """
        Normalizar DataFrame a formato estándar
        """
        df_norm = df.copy()
        
        # Crear columna de fecha si no existe
        if date_col not in df_norm.columns:
            # Intentar crear desde año y semana
            if "ano" in df_norm.columns and "semana" in df_norm.columns:
                df_norm["fecha"] = df_norm.apply(
                    lambda row: self._week_to_date(row["ano"], row["semana"]),
                    axis=1
                )
                date_col = "fecha"
            else:
                raise ValueError("No se pudo determinar la columna de fecha")
        
        # Convertir fecha
        if date_format:
            df_norm[date_col] = pd.to_datetime(df_norm[date_col], format=date_format)
        else:
            df_norm[date_col] = pd.to_datetime(df_norm[date_col], errors='coerce')
        
        # Agregar casos si no existe columna directa
        if case_col is None or case_col not in df_norm.columns:
            # Agregar por ubicación y fecha
            location_key = location_cols + [date_col] if location_cols else [date_col]
            df_norm = df_norm.groupby(location_key).size().reset_index(name="casos")
            case_col = "casos"
        
        # Asegurar columnas de ubicación
        for col in ["departamento", "provincia", "distrito"]:
            if col not in df_norm.columns:
                df_norm[col] = "DESCONOCIDO"
        
        # Crear UBIGEO si no existe
        if "ubigeo" not in df_norm.columns:
            # Intentar crear desde códigos existentes o generar sintético
            df_norm["ubigeo"] = df_norm.get("ubigeo", "000000")
        
        # Agregar features temporales
        df_norm["anio"] = df_norm[date_col].dt.year
        df_norm["mes"] = df_norm[date_col].dt.month
        df_norm["dia_semana"] = df_norm[date_col].dt.weekday
        df_norm["semana_anio"] = df_norm[date_col].dt.isocalendar().week.astype(int)
        df_norm["trimestre"] = df_norm[date_col].dt.quarter
        
        return df_norm.sort_values([*location_cols, date_col])
    
    def _week_to_date(self, year: int, week: int) -> pd.Timestamp:
        """Convertir año-semana a fecha"""
        try:
            # Primer día del año
            first_day = pd.Timestamp(year=year, month=1, day=1)
            # Ajustar al lunes de la semana 1
            first_monday = first_day - pd.Timedelta(days=first_day.weekday())
            # Sumar semanas
            target_date = first_monday + pd.Timedelta(weeks=week-1)
            return target_date
        except:
            return pd.Timestamp(year=year, month=1, day=1)
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        group_by: List[str] = None,
        target_col: str = "casos"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Crear secuencias para entrenamiento
        
        Args:
            df: DataFrame normalizado
            group_by: Columnas para agrupar (default: ubicación)
            target_col: Columna objetivo
            
        Returns:
            X, y, dates, keys
        """
        if group_by is None:
            group_by = ["departamento", "provincia", "distrito"]
        
        if not self.feature_cols:
            # Features por defecto
            self.feature_cols = ["casos", "mes", "dia_semana", "semana_anio", "trimestre"]
        
        X_list, y_list, dates_list, keys_list = [], [], [], []
        
        for keys, group in df.groupby(group_by):
            group = group.sort_values("fecha")
            
            # Asegurar que tenemos las features necesarias
            available_features = [f for f in self.feature_cols if f in group.columns]
            if not available_features:
                continue
            
            values = group[available_features].values
            targets = group[target_col].values
            dates = group["fecha"].values
            
            n = len(group)
            if n <= self.window_size:
                continue
            
            max_i = n - self.window_size - self.horizon + 1
            
            for i in range(max_i):
                x_seq = values[i:i + self.window_size]
                y_val = targets[i + self.window_size:i + self.window_size + self.horizon]
                
                X_list.append(x_seq)
                y_list.append(y_val[0] if len(y_val) > 0 else 0)
                dates_list.append(dates[i + self.window_size + self.horizon - 1])
                keys_list.append(keys)
        
        X = np.array(X_list)
        y = np.array(y_list)
        dates_arr = np.array(dates_list)
        keys_arr = np.array(keys_list, dtype=object)
        
        return X, y, dates_arr, keys_arr
    
    def fit_scaler(self, X: np.ndarray):
        """Ajustar scaler con datos de entrenamiento"""
        n_samples, window_size, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        self.scaler.fit(X_2d)
    
    def transform_sequences(self, X: np.ndarray) -> np.ndarray:
        """Transformar secuencias con scaler"""
        n_samples, window_size, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_2d)
        return X_scaled.reshape(n_samples, window_size, n_features)
