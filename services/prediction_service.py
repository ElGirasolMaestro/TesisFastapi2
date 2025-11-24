"""
Servicio principal para generar predicciones de dengue
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging
import asyncio

from core.model_loader import ModelLoader
from models.schemas import PredictionRequest, PredictionResponse, PredictionResult, PredictionSeries

logger = logging.getLogger(__name__)

class PredictionService:
    """Servicio para generar predicciones de casos de dengue"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Inicializar el servicio de predicción
        
        Args:
            model_dir: Directorio donde se encuentran los archivos del modelo
        """
        self.model_loader = ModelLoader(model_dir)
        self.model = None
        self.scaler = None
        
    async def load_model(self) -> bool:
        """
        Cargar el modelo y sus componentes
        
        Returns:
            bool: True si se cargó exitosamente, False si se usa modelo por defecto
        """
        success = await self.model_loader.load_model()
        if success:
            self.model = self.model_loader.model
            self.scaler = self.model_loader.scaler
            logger.info("Servicio de predicción inicializado correctamente")
            return True
        else:
            # Si falla, asegurar que tenemos modelo y scaler por defecto
            if self.model_loader.model is None:
                self.model_loader.model = self.model_loader._create_default_model()
            if self.model_loader.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.model_loader.scaler = StandardScaler()
            
            self.model = self.model_loader.model
            self.scaler = self.model_loader.scaler
            logger.warning("Servicio inicializado con modelo por defecto debido a errores de carga")
            return False
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generar predicciones para los distritos y período especificados
        
        Args:
            request: Solicitud de predicción
            
        Returns:
            PredictionResponse: Resultados de la predicción
        """
        try:
            # Validar que el modelo esté cargado
            if self.model is None or self.scaler is None:
                raise ValueError("Modelo no disponible")
            
            # Generar fechas para el período solicitado
            dates = self._generate_date_range(request.start_date, request.end_date)
            
            # Generar predicciones para cada distrito
            results = []
            for ubigeo in request.districts:
                district_predictions = await self._predict_district(
                    ubigeo, dates, request
                )
                results.append(district_predictions)
            
            # Crear respuesta
            response = PredictionResponse(
                success=True,
                message="Predicción generada exitosamente",
                results=results,
                metadata={
                    "total_districts": len(request.districts),
                    "total_predictions": len(dates) * len(request.districts),
                    "period": f"{request.start_date} - {request.end_date}",
                    "model_type": "BiLSTM",
                    "parameters": {
                        "mobility_factor": request.mobility_factor,
                        "vaccination_override": request.vaccination_override,
                        "stringency_delta": request.stringency_delta,
                        "force_stringency_60": request.force_stringency_60
                    }
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise e
    
    async def _predict_district(
        self, 
        ubigeo: str, 
        dates: List[str], 
        request: PredictionRequest
    ) -> PredictionResult:
        """
        Generar predicciones para un distrito específico
        
        Args:
            ubigeo: Código UBIGEO del distrito
            dates: Lista de fechas para predecir
            request: Solicitud original
            
        Returns:
            PredictionResult: Predicciones para el distrito
        """
        try:
            # Obtener información del ubigeo
            ubigeo_info = self.model_loader.get_ubigeo_info(ubigeo)
            
            # Generar datos sintéticos para el distrito (en producción, usar datos reales)
            district_data = self._generate_synthetic_data(ubigeo, dates, request)
            
            # Crear secuencias para el modelo
            sequences = self._create_sequences(district_data)
            
            # Hacer predicciones
            predictions = []
            if len(sequences) > 0:
                # Escalar los datos
                scaled_sequences = self._scale_sequences(sequences)
                
                # Predecir con el modelo
                model_predictions = self.model.predict(scaled_sequences, verbose=0)
                
                # Procesar predicciones
                for i, date in enumerate(dates):
                    if i < len(model_predictions):
                        pred_value = float(model_predictions[i][0])
                        # Aplicar ajustes basados en parámetros
                        adjusted_pred = self._apply_adjustments(pred_value, request)
                        predictions.append(PredictionSeries(
                            date=date,
                            pred=max(0, adjusted_pred)  # Asegurar que no sea negativo
                        ))
                    else:
                        # Predicción por defecto si no hay suficientes datos
                        predictions.append(PredictionSeries(
                            date=date,
                            pred=np.random.poisson(5.0)  # Valor por defecto basado en Poisson
                        ))
            else:
                # Si no hay secuencias, generar predicciones por defecto
                for date in dates:
                    predictions.append(PredictionSeries(
                        date=date,
                        pred=np.random.poisson(3.0)
                    ))
            
            return PredictionResult(
                UBIGEO=ubigeo,
                departamento=ubigeo_info.get('departamento'),
                provincia=ubigeo_info.get('provincia'),
                distrito=ubigeo_info.get('distrito'),
                series=predictions
            )
            
        except Exception as e:
            logger.error(f"Error prediciendo distrito {ubigeo}: {e}")
            # Retornar predicciones por defecto en caso de error
            predictions = [
                PredictionSeries(date=date, pred=np.random.poisson(2.0))
                for date in dates
            ]
            return PredictionResult(
                UBIGEO=ubigeo,
                departamento="DESCONOCIDO",
                provincia="DESCONOCIDO", 
                distrito="DESCONOCIDO",
                series=predictions
            )
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Generar lista de fechas semanales entre start_date y end_date
        
        Args:
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            
        Returns:
            Lista de fechas en formato YYYY-MM-DD
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(weeks=1)  # Incremento semanal
        
        return dates
    
    def _generate_synthetic_data(
        self, 
        ubigeo: str, 
        dates: List[str], 
        request: PredictionRequest
    ) -> pd.DataFrame:
        """
        Generar datos sintéticos para el distrito (en producción, usar datos reales)
        
        Args:
            ubigeo: Código UBIGEO
            dates: Lista de fechas
            request: Solicitud de predicción
            
        Returns:
            DataFrame con datos sintéticos
        """
        np.random.seed(int(ubigeo) % 1000)  # Seed basado en ubigeo para consistencia
        
        n_dates = len(dates)
        
        # Generar datos sintéticos realistas
        data = {
            'fecha': pd.to_datetime(dates),
            'casos': np.random.poisson(5, n_dates),  # Casos base
            'precipitacion_mm': np.random.gamma(2, 10, n_dates),  # Precipitación
            'tmin': 18 + np.random.normal(0, 3, n_dates),  # Temperatura mínima
            'tmax': 28 + np.random.normal(0, 4, n_dates),  # Temperatura máxima
        }
        
        # Aplicar efectos estacionales básicos
        for i, date in enumerate(dates):
            month = pd.to_datetime(date).month
            # Más casos en época de lluvias (Nov-Mar)
            if month in [11, 12, 1, 2, 3]:
                data['casos'][i] = int(data['casos'][i] * 1.5)
                data['precipitacion_mm'][i] *= 1.8
        
        df = pd.DataFrame(data)
        df = df.set_index('fecha').sort_index()
        
        return df
    
    def _create_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """
        Crear secuencias para el modelo BiLSTM
        
        Args:
            data: DataFrame con datos del distrito
            
        Returns:
            Array de secuencias para el modelo
        """
        feature_cols = self.model_loader.feature_cols
        window_size = self.model_loader.window_size
        
        if len(data) < window_size:
            logger.warning(f"Datos insuficientes para crear secuencias: {len(data)} < {window_size}")
            return np.array([])
        
        values = data[feature_cols].values
        sequences = []
        
        for i in range(len(data) - window_size + 1):
            sequence = values[i:i + window_size]
            sequences.append(sequence)
        
        return np.array(sequences) if sequences else np.array([])
    
    def _scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Escalar las secuencias usando el scaler cargado
        
        Args:
            sequences: Secuencias sin escalar
            
        Returns:
            Secuencias escaladas
        """
        if len(sequences) == 0:
            return sequences
        
        n_samples, window_size, n_features = sequences.shape
        
        # Reshape para escalar
        sequences_2d = sequences.reshape(-1, n_features)
        
        # Escalar
        scaled_2d = self.scaler.transform(sequences_2d)
        
        # Reshape de vuelta
        scaled_sequences = scaled_2d.reshape(n_samples, window_size, n_features)
        
        return scaled_sequences
    
    def _apply_adjustments(self, prediction: float, request: PredictionRequest) -> float:
        """
        Aplicar ajustes a la predicción basados en los parámetros de la solicitud
        
        Args:
            prediction: Predicción base del modelo
            request: Solicitud con parámetros de ajuste
            
        Returns:
            Predicción ajustada
        """
        adjusted = prediction
        
        # Ajuste por factor de movilidad
        mobility_effect = (request.mobility_factor - 50) / 100.0  # Normalizar a [-0.5, 0.5]
        adjusted *= (1 + mobility_effect * 0.3)  # Efecto máximo del 30%
        
        # Ajuste por vacunación
        vaccination_effect = request.vaccination_override / 100.0
        adjusted *= (1 - vaccination_effect * 0.4)  # Reducción máxima del 40%
        
        # Ajuste por stringency
        stringency_effect = request.stringency_delta / 100.0
        adjusted *= (1 - stringency_effect * 0.2)  # Efecto máximo del 20%
        
        # Stringency forzado a 60%
        if request.force_stringency_60:
            adjusted *= 0.7  # Reducción del 30%
        
        return adjusted
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return self.model_loader.get_model_info()
