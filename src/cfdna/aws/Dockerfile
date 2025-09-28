# Multi-stage Dockerfile for cfDNA cancer detection pipeline
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r cfdna && useradd -r -g cfdna cfdna
RUN mkdir /app && chown cfdna:cfdna /app
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
RUN pip install --upgrade pip

# Production stage
FROM base as production

# Install the package
COPY . .
RUN pip install -e .

# Create directories for data and artifacts
RUN mkdir -p /app/data /app/artifacts /app/models && \
    chown -R cfdna:cfdna /app

# Switch to non-root user
USER cfdna

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cfdna; print('OK')" || exit 1

# Default command shows help
CMD ["python", "-m", "cfdna.train", "--help"]

# Development stage for testing
FROM base as development

# Install development dependencies
COPY . .
RUN pip install -e .[dev]

# Install additional dev tools
RUN pip install jupyterlab ipython

# Create directories
RUN mkdir -p /app/data /app/artifacts /app/models /app/notebooks && \
    chown -R cfdna:cfdna /app

USER cfdna

# Expose Jupyter port
EXPOSE 8888

# Default to Jupyter lab for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]