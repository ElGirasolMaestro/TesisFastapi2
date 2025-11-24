FROM python:3.11-slim

WORKDIR /app

# Instalar librerías esenciales
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo el código necesario
COPY core/ core/
COPY models/ models/
COPY services/ services/
COPY main.py .
COPY config.py .

# Exponer puerto (Cloud Run usará $PORT)
ENV PORT=8080

# Ejecutar la app usando Python (no el binario de uvicorn)
CMD ["python", "main.py"]
