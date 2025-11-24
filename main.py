"""
FastAPI backend para modelos de predicción de enfermedades (Multi-enfermedad con Transfer Learning)
"""
from fastapi import FastAPI, HTTPException, Query, Path, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import uvicorn
import logging
import os
import shutil
import uuid
import json
import zipfile
import tempfile
from io import BytesIO

from services.prediction_service import PredictionService
from services.covid_prediction_service import CovidPredictionService
from services.generic_prediction_service import GenericPredictionService
from services.training_service import TrainingService
from services.notebook_generator import NotebookGenerator
from models.schemas import PredictionRequest, PredictionResponse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Prediction API",
    description="API para predicción de enfermedades usando modelos BiLSTM",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias globales de los servicios de predicción
dengue_service = None
covid_service = None
generic_service = None  # Servicio genérico para múltiples enfermedades
training_service = None  # Servicio de entrenamiento
notebook_generator = None

class DiseaseType(str, Enum):
    """Tipos de enfermedades soportadas (legacy)"""
    DENGUE = "dengue"
    COVID = "covid"

@app.on_event("startup")
async def startup_event():
    """Inicializar los servicios de predicción al arrancar la aplicación"""
    global dengue_service, covid_service, generic_service, training_service, notebook_generator
    
    # Inicializar servicio genérico (soporta múltiples enfermedades)
    try:
        logger.info("Inicializando servicio genérico de predicción...")
        generic_service = GenericPredictionService()
        await generic_service.initialize()
        logger.info("Servicio genérico inicializado exitosamente")
    except Exception as e:
        logger.error(f"Error al inicializar servicio genérico: {e}")
    
    # Inicializar servicio de entrenamiento
    try:
        logger.info("Inicializando servicio de entrenamiento...")
        training_service = TrainingService()
        logger.info("Servicio de entrenamiento inicializado exitosamente")
    except Exception as e:
        logger.error(f"Error al inicializar servicio de entrenamiento: {e}")
    
    # Inicializar generador de notebooks
    try:
        logger.info("Inicializando generador de notebooks...")
        notebook_generator = NotebookGenerator()
        logger.info("Generador de notebooks inicializado exitosamente")
    except Exception as e:
        logger.error(f"Error al inicializar generador de notebooks: {e}")

    # Cargar modelos legacy (para compatibilidad)
    try:
        logger.info("Cargando modelo de predicción de Dengue...")
        dengue_service = PredictionService()
        success = await dengue_service.load_model()
        if success:
            logger.info("Modelo de Dengue cargado exitosamente")
    except Exception as e:
        logger.warning(f"Error al inicializar servicio de Dengue: {e}")
    
    try:
        logger.info("Cargando modelo de predicción de COVID...")
        covid_service = CovidPredictionService()
        success = await covid_service.load_model()
        if success:
            logger.info("Modelo de COVID cargado exitosamente")
    except Exception as e:
        logger.warning(f"Error al inicializar servicio de COVID: {e}")

# Endpoints
@app.get("/")
async def root():
    """Endpoint de salud básico"""
    available_diseases = []
    if generic_service:
        available_diseases = generic_service.get_available_diseases()
    
    return {
        "message": "Disease Prediction API",
        "status": "running",
        "supported_diseases": available_diseases,
        "version": "3.0.0",
        "features": [
            "Multi-enfermedad",
            "Transfer Learning",
            "Predicciones dinámicas",
            "Entrenamiento automático",
            "Generación de notebooks"
        ]
    }

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    available_diseases = []
    disease_status = {}
    if generic_service:
        available_diseases = generic_service.get_available_diseases()
        for disease in available_diseases:
            disease_status[disease] = generic_service.is_loaded(disease)
    
    return {
        "status": "healthy",
        "dengue_model_loaded": dengue_service is not None and dengue_service.model is not None,
        "covid_model_loaded": covid_service is not None and covid_service.model is not None,
        "generic_service_initialized": generic_service is not None,
        "training_service_initialized": training_service is not None,
        "notebook_generator_initialized": notebook_generator is not None,
        "available_diseases": available_diseases,
        "disease_load_status": disease_status
    }

@app.post("/predict/{disease_name}", response_model=PredictionResponse)
async def predict_disease(
    disease_name: str = Path(..., description="Nombre de la enfermedad (ej: dengue, covid, malaria)"),
    request: PredictionRequest = ...
):
    """
    Generar predicciones para cualquier enfermedad disponible
    
    Args:
        disease_name: Nombre de la enfermedad
        request: Datos de entrada para la predicción
        
    Returns:
        PredictionResponse: Resultados de la predicción
    """
    try:
        if generic_service is None:
            raise HTTPException(status_code=503, detail="Servicio genérico no disponible")
        
        # Verificar si la enfermedad está disponible
        available_diseases = generic_service.get_available_diseases()
        if disease_name.lower() not in [d.lower() for d in available_diseases]:
            raise HTTPException(
                status_code=404,
                detail=f"Enfermedad '{disease_name}' no disponible. Enfermedades disponibles: {available_diseases}"
            )
        
        # Normalizar nombre
        disease_name_normalized = next(
            (d for d in available_diseases if d.lower() == disease_name.lower()),
            disease_name
        )
        
        logger.info(f"Generando predicción de {disease_name_normalized} para {len(request.districts)} ubicaciones")
        logger.info(f"Período: {request.start_date} - {request.end_date}")
        
        # Generar predicción usando servicio genérico
        results = await generic_service.predict(disease_name_normalized, request)
        
        logger.info(f"Predicción de {disease_name_normalized} generada exitosamente")
        
        return results
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error interno: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    disease: str = Query("dengue", description="Tipo de enfermedad a predecir")
):
    """
    Generar predicciones de casos (endpoint legacy con query parameter)
    
    Args:
        request: Datos de entrada para la predicción
        disease: Tipo de enfermedad (dengue, covid, malaria, etc.)
        
    Returns:
        PredictionResponse: Resultados de la predicción
    """
    return await predict_disease(disease, request)

@app.post("/predict/dengue", response_model=PredictionResponse)
async def predict_dengue(request: PredictionRequest):
    """
    Generar predicciones de casos de dengue (endpoint específico)
    
    Args:
        request: Datos de entrada para la predicción
        
    Returns:
        PredictionResponse: Resultados de la predicción
    """
    return await predict_disease("dengue", request)

@app.post("/predict/covid", response_model=PredictionResponse)
async def predict_covid(request: PredictionRequest):
    """
    Generar predicciones de casos de COVID (endpoint específico)
    
    Args:
        request: Datos de entrada para la predicción
        
    Returns:
        PredictionResponse: Resultados de la predicción
    """
    return await predict_disease("covid", request)

@app.get("/model/info/{disease_name}")
async def model_info_disease(
    disease_name: str = Path(..., description="Nombre de la enfermedad")
):
    """Información sobre un modelo específico"""
    if generic_service is None:
        raise HTTPException(status_code=503, detail="Servicio genérico no disponible")
    
    try:
        info = generic_service.get_model_info(disease_name)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info(
    disease: Optional[str] = Query(None, description="Tipo de enfermedad")
):
    """Información sobre los modelos cargados"""
    if disease is None:
        # Retornar info de todos los modelos
        if generic_service:
            diseases = generic_service.get_available_diseases()
            return {
                d: generic_service.get_model_info(d)
                for d in diseases
            }
        else:
            return {
                "dengue": dengue_service.get_model_info() if dengue_service else None,
                "covid": covid_service.get_model_info() if covid_service else None
            }
    
    # Info de enfermedad específica
    if generic_service:
        return generic_service.get_model_info(disease)
    else:
        # Fallback a servicios legacy
        if disease.lower() == "dengue" and dengue_service:
            return dengue_service.get_model_info()
        elif disease.lower() == "covid" and covid_service:
            return covid_service.get_model_info()
        else:
            raise HTTPException(status_code=404, detail=f"Modelo de {disease} no disponible")

@app.get("/diseases")
async def list_diseases():
    """Listar todas las enfermedades disponibles"""
    if generic_service is None:
        raise HTTPException(status_code=503, detail="Servicio genérico no disponible")
    
    diseases = generic_service.get_available_diseases()
    disease_info = {}
    
    for disease in diseases:
        try:
            disease_info[disease] = generic_service.get_model_info(disease)
        except Exception as e:
            disease_info[disease] = {"error": str(e)}
    
    return {
        "available_diseases": diseases,
        "disease_info": disease_info
    }

@app.post("/train/upload")
async def upload_and_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    disease_name: str = Query(..., description="Nombre de la enfermedad"),
    window_size: int = Query(8, description="Tamaño de ventana temporal"),
    horizon: int = Query(1, description="Horizonte de predicción"),
    epochs: int = Query(20, description="Número de épocas"),
    batch_size: int = Query(64, description="Tamaño de batch"),
    base_model: Optional[str] = Query(None, description="Modelo base (opcional)")
):
    """
    Subir CSV y entrenar modelo automáticamente
    
    Args:
        file: Archivo CSV con datos
        disease_name: Nombre de la enfermedad
        window_size: Tamaño de ventana temporal
        horizon: Horizonte de predicción
        epochs: Número de épocas
        batch_size: Tamaño de batch
        base_model: Modelo base para transfer learning (opcional)
        
    Returns:
        Job ID del entrenamiento
    """
    if training_service is None:
        raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
    
    try:
        # Crear directorio de uploads si no existe
        os.makedirs("uploads", exist_ok=True)
        
        # Guardar archivo temporalmente
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join("uploads", safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Archivo guardado: {file_path}")
        
        # Iniciar entrenamiento en background
        result = await training_service.train_from_csv(
            csv_path=file_path,
            disease_name=disease_name,
            window_size=window_size,
            horizon=horizon,
            epochs=epochs,
            batch_size=batch_size,
            base_model_path=base_model
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str = Path(..., description="ID del trabajo de entrenamiento")):
    """
    Obtener estado de un trabajo de entrenamiento
    
    Args:
        job_id: ID del trabajo
        
    Returns:
        Estado del trabajo
    """
    if training_service is None:
        raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
    
    try:
        status = training_service.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/jobs")
async def list_training_jobs():
    """
    Listar todos los trabajos de entrenamiento
    
    Returns:
        Lista de trabajos
    """
    if training_service is None:
        raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
    
    try:
        jobs = training_service.list_jobs()
        return jobs
    except Exception as e:
        logger.error(f"Error listando trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/reload")
async def reload_models():
    """
    Recargar todos los modelos disponibles
    
    Returns:
        Estado de recarga
    """
    if generic_service is None:
        raise HTTPException(status_code=503, detail="Servicio de predicción no disponible")
    
    try:
        # Recargar todos los modelos
        await generic_service.model_manager.reload_all_models()
        
        # Obtener estado de todas las enfermedades
        diseases = await generic_service.get_available_diseases()
        
        return {
            "message": "Modelos recargados",
            "diseases": diseases
        }
    except Exception as e:
        logger.error(f"Error recargando modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/base-models")
async def get_base_models():
    """
    Obtener lista de modelos base disponibles para transfer learning
    
    Returns:
        Lista de modelos base
    """
    if generic_service is None:
        raise HTTPException(status_code=503, detail="Servicio de predicción no disponible")
    
    try:
        models_dir = "models"
        base_models = []
        
        # Modelos base conocidos
        known_models = {
            "dengue": {
                "id": "dengue",
                "name": "Dengue (Modelo Base)",
                "path": os.path.join(models_dir, "bilstm_model.h5"),
                "description": "Modelo BiLSTM entrenado para predicción de dengue",
                "available": os.path.exists(os.path.join(models_dir, "bilstm_model.h5"))
            },
            "covid": {
                "id": "covid",
                "name": "COVID-19 (Modelo Base)",
                "path": os.path.join(models_dir, "bilstm_covid_provincial.h5"),
                "description": "Modelo BiLSTM entrenado para predicción de COVID-19",
                "available": os.path.exists(os.path.join(models_dir, "bilstm_covid_provincial.h5"))
            }
        }
        
        for model_id, model_info in known_models.items():
            if model_info["available"]:
                base_models.append(model_info)
        
        # Buscar modelos entrenados dinámicamente
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".h5"):
                    model_name = file.replace("bilstm_", "").replace(".h5", "")
                    
                    # Excluir modelos base conocidos
                    if model_name not in ["model", "covid_provincial"]:
                        config_file = os.path.join(models_dir, f"config_{model_name}.json")
                        model_info = {
                            "id": model_name,
                            "name": f"{model_name.capitalize()} (Modelo Entrenado)",
                            "path": os.path.join(models_dir, file),
                            "description": f"Modelo entrenado para {model_name} usando transfer learning",
                            "available": True
                        }
                        
                        # Cargar info del config si existe
                        if os.path.exists(config_file):
                            try:
                                with open(config_file, 'r') as f:
                                    config = json.load(f)
                                model_info.update({
                                    "features": config.get("features", []),
                                    "window_size": config.get("window_size", 8),
                                    "metrics": config.get("metrics", {})
                                })
                            except Exception as e:
                                logger.warning(f"Error cargando config para {model_name}: {e}")
                        
                        base_models.append(model_info)
        
        return {
            "total_models": len(base_models),
            "models": base_models
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notebook/{disease_name}")
async def download_notebook(
    disease_name: str = Path(..., description="Nombre de la enfermedad")
):
    """
    Generar y descargar notebook Jupyter de análisis para una enfermedad
    Incluye el notebook, modelo (.h5), scaler (.pkl) y config (.json) en un ZIP
    
    Args:
        disease_name: Nombre de la enfermedad (dengue, covid, malaria, etc.)
        
    Returns:
        Archivo ZIP descargable con notebook y archivos del modelo
    """
    if notebook_generator is None:
        raise HTTPException(status_code=503, detail="Generador de notebooks no disponible")
    
    try:
        models_dir = "models"
        
        # Determinar nombres de archivos según la enfermedad
        if disease_name.lower() == "dengue":
            notebook = notebook_generator.generate_dengue_notebook()
            notebook_filename = "analisis_modelo_dengue.ipynb"
            model_file = "bilstm_model.h5"
            scaler_file = "scaler.pkl"
            config_file = None
            zip_filename = "analisis_modelo_dengue.zip"
        elif disease_name.lower() == "covid":
            notebook = notebook_generator.generate_covid_notebook()
            notebook_filename = "analisis_modelo_covid.ipynb"
            model_file = "bilstm_covid_provincial.h5"
            scaler_file = "scaler_covid_provincial.pkl"
            config_file = None
            zip_filename = "analisis_modelo_covid.zip"
        else:
            # Notebook genérico
            config = None
            config_path = os.path.join(models_dir, f"config_{disease_name}.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"No se pudo cargar config para {disease_name}: {e}")
            
            notebook = notebook_generator.generate_generic_notebook(disease_name, config)
            notebook_filename = f"analisis_modelo_{disease_name}.ipynb"
            model_file = f"bilstm_{disease_name}.h5"
            scaler_file = f"scaler_{disease_name}.pkl"
            config_file = f"config_{disease_name}.json"
            zip_filename = f"analisis_modelo_{disease_name}.zip"
        
        # Crear ZIP en memoria
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Agregar notebook
            notebook_json = json.dumps(notebook, indent=1, ensure_ascii=False)
            zip_file.writestr(notebook_filename, notebook_json.encode('utf-8'))
            
            # Agregar modelo (.h5)
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                zip_file.write(model_path, model_file)
                logger.info(f"✅ Modelo agregado al ZIP: {model_file}")
            else:
                logger.warning(f"⚠️ Modelo no encontrado: {model_path}")
            
            # Agregar scaler (.pkl)
            scaler_path = os.path.join(models_dir, scaler_file)
            if os.path.exists(scaler_path):
                zip_file.write(scaler_path, scaler_file)
                logger.info(f"✅ Scaler agregado al ZIP: {scaler_file}")
            else:
                logger.warning(f"⚠️ Scaler no encontrado: {scaler_path}")
            
            # Agregar config (.json) si existe
            if config_file:
                config_path = os.path.join(models_dir, config_file)
                if os.path.exists(config_path):
                    zip_file.write(config_path, config_file)
                    logger.info(f"✅ Config agregado al ZIP: {config_file}")
        
        # Preparar respuesta
        zip_buffer.seek(0)
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "Content-Type": "application/zip"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generando notebook para {disease_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando notebook: {str(e)}")

@app.get("/notebooks/available")
async def get_available_notebooks():
    """
    Obtener lista de enfermedades para las que se pueden generar notebooks
    
    Returns:
        Lista de enfermedades disponibles
    """
    if notebook_generator is None:
        raise HTTPException(status_code=503, detail="Generador de notebooks no disponible")
    
    try:
        diseases = ["dengue", "covid"]
        
        # Agregar enfermedades con modelos entrenados
        models_dir = "models"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.startswith("bilstm_") and file.endswith(".h5"):
                    disease_name = file.replace("bilstm_", "").replace(".h5", "")
                    if disease_name not in ["model", "covid_provincial"] and disease_name not in diseases:
                        diseases.append(disease_name)
        
        return {
            "available_diseases": diseases,
            "total": len(diseases)
        }
    except Exception as e:
        logger.error(f"Error obteniendo notebooks disponibles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)



