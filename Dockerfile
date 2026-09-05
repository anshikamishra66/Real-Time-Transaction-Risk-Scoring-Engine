FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build a demo-ready dataset and train models at image-build time so the
# container is immediately runnable with zero setup steps -- no Kaggle
# credentials required (see data/download_data.py for using the real dataset).
RUN python scripts/generate_synthetic_data.py --n-rows 250000 \
    && python data/preprocess.py \
    && python -m models.train_all

EXPOSE 8000 8501

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 & python -m streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & wait"]
