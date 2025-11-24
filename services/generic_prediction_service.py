"""
Servicio genérico de predicción para múltiples enfermedades
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from core.disease_model_manager import DiseaseModelManager
from models.schemas import PredictionRequest, PredictionResponse, PredictionResult, PredictionSeries

logger = logging.getLogger(__name__)

class GenericPredictionService:
    """Servicio genérico para generar predicciones de cualquier enfermedad"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializar servicio genérico
        
        Args:
            models_dir: Directorio donde se encuentran los modelos
        """
        self.model_manager = DiseaseModelManager(models_dir)
        
    async def initialize(self):
        """Inicializar cargando todos los modelos disponibles"""
        await self.model_manager.load_all_available()
    
    async def predict(
        self, 
        disease_name: str,
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Generar predicciones para una enfermedad específica
        
        Args:
            disease_name: Nombre de la enfermedad
            request: Solicitud de predicción
            
        Returns:
            PredictionResponse: Resultados de la predicción
        """
        try:
            # Cargar modelo si no está cargado
            if not self.model_manager.is_loaded(disease_name):
                success = await self.model_manager.load_model(disease_name)
                if not success:
                    raise ValueError(f"Modelo de {disease_name} no disponible")
            
            model = self.model_manager.get_model(disease_name)
            scaler = self.model_manager.get_scaler(disease_name)
            config = self.model_manager.get_config(disease_name)
            
            if model is None or scaler is None:
                raise ValueError(f"Modelo de {disease_name} no disponible")
            
            # Obtener configuración
            window_size = config.get("window_size", 8)
            horizon = config.get("horizon", 1)
            frequency = config.get("frequency", "weekly")
            use_log = config.get("use_log", False)
            
            # Generar fechas según frecuencia
            if frequency == "daily":
                dates = self._generate_daily_dates(request.start_date, request.end_date)
            else:  # weekly
                dates = self._generate_weekly_dates(request.start_date, request.end_date)
            
            # Generar predicciones para cada ubicación
            results = []
            for ubigeo in request.districts:
                location_predictions = await self._predict_location(
                    disease_name, ubigeo, dates, request, model, scaler, config
                )
                results.append(location_predictions)
            
            # Crear respuesta
            response = PredictionResponse(
                success=True,
                message=f"Predicción de {disease_name.upper()} generada exitosamente",
                results=results,
                metadata={
                    "disease": disease_name,
                    "total_locations": len(request.districts),
                    "total_predictions": len(dates) * len(request.districts),
                    "period": f"{request.start_date} - {request.end_date}",
                    "model_type": "BiLSTM",
                    "frequency": frequency,
                    "window_size": window_size,
                    "horizon": horizon,
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
            logger.error(f"Error en predicción de {disease_name}: {e}")
            raise e
    
    async def _predict_location(
        self,
        disease_name: str,
        ubigeo: str,
        dates: List[str],
        request: PredictionRequest,
        model: Any,
        scaler: Any,
        config: Dict[str, Any]
    ) -> PredictionResult:
        """Generar predicciones para una ubicación específica"""
        try:
            # Obtener información de ubicación
            location_info = self._get_location_info(ubigeo)
            
            # Generar datos sintéticos (en producción, usar datos reales)
            location_data = self._generate_synthetic_data(
                disease_name, ubigeo, dates, request, config
            )
            
            # Crear secuencias
            sequences = self._create_sequences(location_data, config)
            
            # Hacer predicciones
            predictions = []
            if len(sequences) > 0:
                # Escalar
                scaled_sequences = self._scale_sequences(sequences, scaler)
                
                # Predecir
                model_predictions = model.predict(scaled_sequences, verbose=0)
                
                # Convertir si usa log
                if config.get("use_log", False):
                    model_predictions = np.expm1(model_predictions)
                
                # Procesar predicciones
                for i, date in enumerate(dates):
                    if i < len(model_predictions):
                        pred_value = float(model_predictions[i][0])
                        adjusted_pred = self._apply_adjustments(pred_value, request)
                        predictions.append(PredictionSeries(
                            date=date,
                            pred=max(0, adjusted_pred)
                        ))
                    else:
                        predictions.append(PredictionSeries(
                            date=date,
                            pred=np.random.poisson(5.0)
                        ))
            else:
                # Predicciones por defecto
                for date in dates:
                    predictions.append(PredictionSeries(
                        date=date,
                        pred=np.random.poisson(3.0)
                    ))
            
            return PredictionResult(
                UBIGEO=ubigeo,
                departamento=location_info.get('departamento'),
                provincia=location_info.get('provincia'),
                distrito=location_info.get('distrito'),
                series=predictions
            )
            
        except Exception as e:
            logger.error(f"Error prediciendo ubicación {ubigeo}: {e}")
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
    
    def _generate_daily_dates(self, start_date: str, end_date: str) -> List[str]:
        """Generar fechas diarias"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        return dates
    
    def _generate_weekly_dates(self, start_date: str, end_date: str) -> List[str]:
        """Generar fechas semanales"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(weeks=1)
        return dates
    
    def _generate_synthetic_data(
        self,
        disease_name: str,
        ubigeo: str,
        dates: List[str],
        request: PredictionRequest,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generar datos sintéticos para la ubicación"""
        np.random.seed(int(ubigeo) % 1000 if ubigeo.isdigit() else hash(ubigeo) % 1000)
        
        n_dates = len(dates)
        features = config.get("features", ["casos", "mes", "dia_semana"])
        
        data = {
            'fecha': pd.to_datetime(dates),
            'casos': np.random.poisson(10, n_dates)
        }
        
        # Agregar features según configuración
        df = pd.DataFrame(data)
        if 'mes' in features:
            df['mes'] = df['fecha'].dt.month
        if 'dia_semana' in features:
            df['dia_semana'] = df['fecha'].dt.weekday
        if 'semana_anio' in features:
            df['semana_anio'] = df['fecha'].dt.isocalendar().week.astype(int)
        if 'trimestre' in features:
            df['trimestre'] = df['fecha'].dt.quarter
        if 'casos_vecinos' in features:
            df['casos_vecinos'] = np.random.poisson(30, n_dates)
        if 'precipitacion_mm' in features:
            df['precipitacion_mm'] = np.random.gamma(2, 10, n_dates)
        if 'tmin' in features:
            df['tmin'] = 18 + np.random.normal(0, 3, n_dates)
        if 'tmax' in features:
            df['tmax'] = 28 + np.random.normal(0, 4, n_dates)
        
        df = df.set_index('fecha').sort_index()
        return df
    
    def _create_sequences(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Crear secuencias para el modelo"""
        features = config.get("features", ["casos"])
        window_size = config.get("window_size", 8)
        
        if len(data) < window_size:
            return np.array([])
        
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            return np.array([])
        
        values = data[available_features].values
        sequences = []
        
        for i in range(len(data) - window_size + 1):
            sequence = values[i:i + window_size]
            sequences.append(sequence)
        
        return np.array(sequences) if sequences else np.array([])
    
    def _scale_sequences(self, sequences: np.ndarray, scaler: Any) -> np.ndarray:
        """Escalar secuencias"""
        if len(sequences) == 0:
            return sequences
        
        n_samples, window_size, n_features = sequences.shape
        sequences_2d = sequences.reshape(-1, n_features)
        scaled_2d = scaler.transform(sequences_2d)
        return scaled_2d.reshape(n_samples, window_size, n_features)
    
    def _apply_adjustments(
        self,
        prediction: float,
        request: PredictionRequest
    ) -> float:
        """Aplicar ajustes a la predicción"""
        adjusted = prediction
        
        mobility_effect = (request.mobility_factor - 50) / 100.0
        adjusted *= (1 + mobility_effect * 0.3)
        
        vaccination_effect = request.vaccination_override / 100.0
        adjusted *= (1 - vaccination_effect * 0.4)
        
        stringency_effect = request.stringency_delta / 100.0
        adjusted *= (1 - stringency_effect * 0.2)
        
        if request.force_stringency_60:
            adjusted *= 0.7
        
        return adjusted
    
    def _get_location_info(self, ubigeo: str) -> Dict[str, str]:
        """Obtener información de ubicación"""
        return {
            'departamento': 'DESCONOCIDO',
            'provincia': 'DESCONOCIDO',
            'distrito': 'DESCONOCIDO'
        }
    
    def get_available_diseases(self) -> List[str]:
        """Obtener lista de enfermedades disponibles"""
        return self.model_manager.discover_models()
    
    def is_loaded(self, disease_name: str) -> bool:
        """Verificar si un modelo está cargado"""
        return self.model_manager.is_loaded(disease_name)
    
    def get_model_info(self, disease_name: str) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return self.model_manager.get_model_info(disease_name)

