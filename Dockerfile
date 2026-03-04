# ==========================================
# Food Recommendation API — Dockerfile
# ==========================================
# Multi-purpose Docker image:
# 1. Installs Python dependencies
# 2. Copies the trained model + API code
# 3. Runs FastAPI with Uvicorn on port 8000
#
# Build:  docker build -t food-recommender .
# Run:    docker run -p 8000:8000 food-recommender
# Test:   curl http://localhost:8000/health
# ==========================================

FROM python:3.11-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files (cleaner container)
# PYTHONUNBUFFERED: Ensures logs appear in real-time in Docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install dependencies first (Docker caches this layer)
# If requirements.txt doesn't change, Docker skips this step on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/

# Expose the API port
EXPOSE 8000

# Health check — Kubernetes also does this, but Docker health check is useful for local testing
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
# --host 0.0.0.0: Listen on all interfaces (required in containers)
# --port 8000: Match the EXPOSE above
# --workers 2: Number of worker processes (adjust based on CPU)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
