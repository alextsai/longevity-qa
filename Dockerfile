FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --only-binary=:all: -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["streamlit","run","app/app.py","--server.port=8080","--server.address=0.0.0.0"]
