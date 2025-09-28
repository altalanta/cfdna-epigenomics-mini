# cfDNA Epigenomics Mini - Production Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy and install Python dependencies
COPY --chown=app:app pyproject.toml ./
RUN pip install --user --no-cache-dir -e .[viz]

# Copy application code
COPY --chown=app:app . .

# Install the package
RUN pip install --user --no-cache-dir -e .

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:$PATH"

# Create working directory for data
RUN mkdir -p /home/app/workdir
WORKDIR /home/app/workdir

# Default command - run smoke test
CMD ["cfdna", "smoke", "--seed", "42"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD cfdna --version || exit 1

# Labels
LABEL maintainer="cfDNA Team" \
      version="0.1.0" \
      description="Early cancer detection from cfDNA epigenomic signals" \
      repository="https://github.com/example/cfdna-epigenomics-mini"