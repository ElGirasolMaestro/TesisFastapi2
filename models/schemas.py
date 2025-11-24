"""
Schemas Pydantic para la API de predicción
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

class PredictionRequest(BaseModel):
    """Esquema para solicitud de predicción"""
    
    districts: List[str] = Field(
        ..., 
        description="Lista de códigos UBIGEO de distritos",
        min_items=1,
        example=["150101", "150102", "150103"]
    )
    
    start_date: str = Field(
        ...,
        description="Fecha de inicio en formato YYYY-MM-DD",
        example="2024-01-01"
    )
    
    end_date: str = Field(
        ...,
        description="Fecha de fin en formato YYYY-MM-DD", 
        example="2024-12-31"
    )
    
    mobility_factor: int = Field(
        default=1,
        description="Factor de movilidad (entero)",
        ge=0,
        le=100,
        example=75
    )
    
    vaccination_override: int = Field(
        default=0,
        description="Override de vacunación (%)",
        ge=0,
        le=100,
        example=80
    )
    
    stringency_delta: int = Field(
        default=0,
        description="Delta de stringency",
        ge=-50,
        le=50,
        example=10
    )
    
    force_stringency_60: bool = Field(
        default=False,
        description="Forzar stringency a 60%",
        example=False
    )
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validar formato de fecha YYYY-MM-DD"""
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError('Formato de fecha debe ser YYYY-MM-DD')
        
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Fecha inválida')
        
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validar que end_date sea posterior a start_date"""
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            
            if end <= start:
                raise ValueError('end_date debe ser posterior a start_date')
        
        return v
    
    @validator('districts')
    def validate_districts(cls, v):
        """Validar códigos UBIGEO"""
        for district in v:
            if not isinstance(district, str):
                # Permitir códigos no numéricos (como "LIMA|LIMA" para COVID)
                continue
            if len(district) != 6 and district.isdigit():
                raise ValueError(f'Código UBIGEO inválido: {district}. Debe tener 6 dígitos')
        
        return v

class PredictionSeries(BaseModel):
    """Serie temporal de predicciones para un distrito"""
    
    date: str = Field(..., description="Fecha de la predicción")
    pred: float = Field(..., description="Valor predicho de casos")

class PredictionResult(BaseModel):
    """Resultado de predicción para un distrito"""
    
    UBIGEO: str = Field(..., description="Código UBIGEO del distrito")
    departamento: Optional[str] = Field(None, description="Nombre del departamento")
    provincia: Optional[str] = Field(None, description="Nombre de la provincia") 
    distrito: Optional[str] = Field(None, description="Nombre del distrito")
    series: List[PredictionSeries] = Field(..., description="Serie temporal de predicciones")

class PredictionResponse(BaseModel):
    """Respuesta de la API de predicción"""
    
    success: bool = Field(True, description="Indica si la predicción fue exitosa")
    message: str = Field("Predicción generada exitosamente", description="Mensaje descriptivo")
    results: List[PredictionResult] = Field(..., description="Resultados de predicción por distrito")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos adicionales sobre la predicción"
    )

class ModelInfo(BaseModel):
    """Información del modelo"""
    disease_name: str = Field(..., description="Nombre de la enfermedad")
    model_type: str = Field(..., description="Tipo de modelo")
    window_size: int = Field(..., description="Tamaño de ventana temporal")
    horizon: int = Field(..., description="Horizonte de predicción")
    features: List[str] = Field(..., description="Features utilizadas")
    frequency: str = Field(..., description="Frecuencia de los datos")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    scaler_loaded: bool = Field(..., description="Si el scaler está cargado")
    input_shape: Optional[str] = Field(None, description="Shape de entrada")
    output_shape: Optional[str] = Field(None, description="Shape de salida")

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    error: str = Field(..., description="Mensaje de error")
    detail: Optional[str] = Field(None, description="Detalle del error")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp del error")
