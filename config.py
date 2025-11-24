"""
Configuración de la aplicación
"""
import os
from typing import Optional

class Settings:
    """Configuración de la aplicación"""
    
    # Configuración del servidor
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Configuración del modelo
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    MODEL_FILE: str = os.getenv("MODEL_FILE", "bilstm_model.h5")
    SCALER_FILE: str = os.getenv("SCALER_FILE", "scaler.pkl")
    UBIGEOS_FILE: str = os.getenv("UBIGEOS_FILE", "ubigeos.csv")
    
    # Configuración de logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Configuración de CORS
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Configuración del modelo
    WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", "8"))
    HORIZON: int = int(os.getenv("HORIZON", "1"))
    FEATURE_COLS: list = ["casos", "precipitacion_mm", "tmin", "tmax"]
    
    # Límites de la API
    MAX_DISTRICTS: int = int(os.getenv("MAX_DISTRICTS", "100"))
    MAX_DATE_RANGE_DAYS: int = int(os.getenv("MAX_DATE_RANGE_DAYS", "365"))
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.MODEL_DIR, self.MODEL_FILE)
    
    @property
    def scaler_path(self) -> str:
        return os.path.join(self.MODEL_DIR, self.SCALER_FILE)
    
    @property
    def ubigeos_path(self) -> str:
        return os.path.join(self.MODEL_DIR, self.UBIGEOS_FILE)

# Instancia global de configuración
settings = Settings()

