FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Upgrade pip and install dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Cloud Run provides the PORT environment variable dynamically (defaults to 8080)
ENV PORT=8080

# Start the uvicorn server using the port Cloud Run gives us
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
