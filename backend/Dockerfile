# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies and curl for uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (faster pip replacement)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project metadata first (for caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies directly into system Python (no venv)
RUN uv pip install --system --no-cache-dir -r <(uv pip compile --extra prod pyproject.toml)

# Copy app code
COPY . .

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed deps from builder (system site-packages)
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV CORS_ORIGINS=["https://appsageai.com","https://appsageai.com"]
ENV ALLOWED_FILE_TYPES=["pdf"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]