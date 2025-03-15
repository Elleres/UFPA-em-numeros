FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    poppler-utils \
    libpoppler-cpp-dev \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Configurar o comando de execução do uvicorn diretamente no Dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
