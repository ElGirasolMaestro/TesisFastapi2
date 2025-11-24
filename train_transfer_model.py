"""
Script para entrenar modelos de nuevas enfermedades usando Transfer Learning
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import joblib
import json

from core.transfer_learning import TransferLearningTrainer, DiseaseDataPreprocessor

def train_new_disease_model(
    csv_path: str,
    disease_name: str,
    base_model_path: str = "models/bilstm_model.h5",  # Modelo dengue como base
    output_dir: str = "models",
    window_size: int = 8,
    horizon: int = 1,
    test_split_date: str = None,
    epochs: int = 20,
    batch_size: int = 64
):
    """
    Entrenar modelo para nueva enfermedad usando transfer learning
    
    Args:
        csv_path: Ruta al CSV de la nueva enfermedad
        disease_name: Nombre de la enfermedad (ej: "malaria")
        base_model_path: Ruta al modelo base (dengue o covid)
        output_dir: Directorio para guardar el modelo entrenado
        window_size: Tamaño de ventana temporal
        horizon: Horizonte de predicción
        test_split_date: Fecha para split train/test (YYYY-MM-DD)
        epochs: Número de épocas
        batch_size: Tamaño de batch
    """
    print(f"🚀 Entrenando modelo para {disease_name} usando Transfer Learning")
    print(f"📂 CSV: {csv_path}")
    print(f"📦 Modelo base: {base_model_path}")
    
    # 1. Preprocesar datos
    print("\n📊 Preprocesando datos...")
    preprocessor = DiseaseDataPreprocessor(
        window_size=window_size,
        horizon=horizon
    )
    
    # Cargar y normalizar CSV
    df = preprocessor.load_csv(csv_path)
    
    # Crear secuencias
    X, y, dates, keys = preprocessor.create_sequences(df)
    print(f"✅ Secuencias creadas: {X.shape}")
    
    # 2. Split train/test
    if test_split_date:
        split_date = pd.to_datetime(test_split_date)
        mask_test = dates >= split_date
        mask_train = ~mask_test
    else:
        # Split 80/20 por defecto
        split_idx = int(0.8 * len(X))
        mask_train = np.array([True] * split_idx + [False] * (len(X) - split_idx))
        mask_test = ~mask_train
    
    X_train, X_test = X[mask_train], X[mask_test]
    y_train, y_test = y[mask_train], y[mask_test]
    
    print(f"📈 Train: {X_train.shape[0]} muestras")
    print(f"📉 Test: {X_test.shape[0]} muestras")
    
    # 3. Escalar datos
    print("\n🔧 Escalando datos...")
    preprocessor.fit_scaler(X_train)
    X_train_scaled = preprocessor.transform_sequences(X_train)
    X_test_scaled = preprocessor.transform_sequences(X_test)
    
    # Transformar target a log si es necesario (para valores grandes)
    if y_train.max() > 100:
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        use_log = True
    else:
        y_train_log = y_train
        y_test_log = y_test
        use_log = False
    
    # 4. Crear modelo de transfer learning
    print("\n🧠 Creando modelo de transfer learning...")
    trainer = TransferLearningTrainer(base_model_path=base_model_path)
    transfer_model = trainer.create_transfer_model(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        freeze_encoder=True
    )
    
    print(f"📋 Arquitectura del modelo:")
    transfer_model.summary()
    
    # 5. Entrenar
    print("\n🚂 Entrenando modelo...")
    trained_model = trainer.train_transfer_model(
        X_train_scaled,
        y_train_log,
        X_test_scaled,
        y_test_log,
        epochs=epochs,
        batch_size=batch_size,
        patience=5
    )
    
    # 6. Evaluar
    print("\n📊 Evaluando modelo...")
    y_pred_log = trained_model.predict(X_test_scaled).ravel()
    y_pred = np.expm1(y_pred_log) if use_log else y_pred_log
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ MAE: {mae:.3f}")
    print(f"✅ RMSE: {rmse:.3f}")
    print(f"✅ R²: {r2:.3f}")
    
    # 7. Guardar modelo y componentes
    print("\n💾 Guardando modelo...")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"bilstm_{disease_name}.h5")
    scaler_path = os.path.join(output_dir, f"scaler_{disease_name}.pkl")
    config_path = os.path.join(output_dir, f"config_{disease_name}.json")
    
    trained_model.save(model_path)
    joblib.dump(preprocessor.scaler, scaler_path)
    
    # Guardar configuración
    config = {
        "disease_name": disease_name,
        "window_size": window_size,
        "horizon": horizon,
        "features": preprocessor.feature_cols,
        "base_model": base_model_path,
        "use_log": use_log,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "trained_date": datetime.now().isoformat(),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0])
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Modelo guardado en: {model_path}")
    print(f"✅ Scaler guardado en: {scaler_path}")
    print(f"✅ Config guardado en: {config_path}")
    
    print(f"\n🎉 Modelo de {disease_name} entrenado exitosamente!")
    print(f"\nPara usar en la API, agrega el modelo al sistema de gestión de enfermedades.")
    
    return trained_model, preprocessor, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo con Transfer Learning")
    parser.add_argument("--csv", required=True, help="Ruta al CSV de la enfermedad")
    parser.add_argument("--disease", required=True, help="Nombre de la enfermedad")
    parser.add_argument("--base-model", default="models/bilstm_model.h5", help="Modelo base")
    parser.add_argument("--output-dir", default="models", help="Directorio de salida")
    parser.add_argument("--window-size", type=int, default=8, help="Tamaño de ventana")
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte de predicción")
    parser.add_argument("--test-date", help="Fecha de split test (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de batch")
    
    args = parser.parse_args()
    
    train_new_disease_model(
        csv_path=args.csv,
        disease_name=args.disease,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        window_size=args.window_size,
        horizon=args.horizon,
        test_split_date=args.test_date,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

