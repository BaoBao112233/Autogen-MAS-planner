# MAS-Planning Multi-Agent System Dockerfile
# Optimized for development and production deployment

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy Poetry files first for Docker layer caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install Python dependencies using Poetry
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs data mcp_config mcp_data

# Create non-root user for security
RUN groupadd -r mas && useradd -r -g mas mas && \
    chown -R mas:mas /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Switch to non-root user
USER mas

# Production command (use --reload for development)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--log-level", "info"]
