FROM python:3.11-slim

# Opciones útiles de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Paquetes de sistema básicos (para compilar dependencias)
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo el código necesario
COPY core/ core/
COPY models/ models/
COPY services/ services/
COPY storage/ storage/
COPY main.py .
COPY config.py .

# Nombre del bucket (puedes sobreescribirlo en Cloud Run)
ENV GCS_BUCKET_NAME=tesis-datasets

# Exponer puerto (Cloud Run usa $PORT)
ENV PORT=8080

# Ejecutar la app
CMD ["python", "main.py"]
