# Dockerfile
FROM python:3.13-slim

# ---- Runtime env (stable + low memory) ----
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1 \
    DATA_DIR=/var/data \
    SENTENCE_TRANSFORMERS_HOME=/var/data/models \
    HF_HOME=/var/data/models

# ---- System deps (lean) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps first (layer cache) ----
COPY requirements.txt .
# Prefer wheels to avoid slow source builds (works for torch, faiss-cpu on py3.13)
RUN python -m pip install --upgrade pip && \
    pip install --only-binary=:all: -r requirements.txt

# ---- App code (includes scripts/bootstrap_data.py) ----
COPY . .

# ---- Streamlit port for Railway ----
EXPOSE 8080

# ---- Boot: fetch data into /var/data, build offsets, then start app ----
CMD ["bash","-lc","which python || true; python -V || true; export DATA_DIR=/var/data; python scripts/bootstrap_data.py || true; python -m streamlit run app/app.py --server.port=8080 --server.address=0.0.0.0"]
