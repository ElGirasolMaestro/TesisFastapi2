"""
Módulo de modelos de datos para la API de predicción
"""
from .schemas import (
    PredictionRequest,
    PredictionResponse, 
    PredictionResult,
    PredictionSeries,
    ModelInfo,
    ErrorResponse
)

__all__ = [
    "PredictionRequest",
    "PredictionResponse",
    "PredictionResult", 
    "PredictionSeries",
    "ModelInfo",
    "ErrorResponse"
]

