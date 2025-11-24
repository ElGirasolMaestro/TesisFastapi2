"""
Módulo de servicios para la API de predicción
"""
from .prediction_service import PredictionService
from .covid_prediction_service import CovidPredictionService
from .generic_prediction_service import GenericPredictionService

__all__ = [
    "PredictionService",
    "CovidPredictionService",
    "GenericPredictionService"
]
