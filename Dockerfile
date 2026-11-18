FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY data ./data

ENV PYTHONPATH=/app/src:/app
# Train inside the image so the model always matches the library versions installed here.
RUN python scripts/train.py --no-backtest --no-mlflow

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
