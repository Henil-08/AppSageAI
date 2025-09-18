# Build stage
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml .
RUN uv pip install -r pyproject.toml

COPY . .
RUN uv pip install -e .

# Expose port
EXPOSE 8000

ENV CORS_ORIGINS=["https://appsageai.com","https://appsageai.com"]
ENV ALLOWED_FILE_TYPES=["pdf"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]