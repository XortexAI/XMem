FROM python:3.11-slim

# Install system dependencies (git is required for the scanner to clone repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Upgrade pip and install dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Cloud Run provides the PORT environment variable dynamically (defaults to 8080)
ENV PORT=8080

# Use the same entrypoint as the Procfile
CMD uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}
