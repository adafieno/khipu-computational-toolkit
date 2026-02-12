# Khipu Cloud Viewer - Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cloud_viewer/ ./cloud_viewer/
COPY data/blob_export/ ./data/blob_export/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STORAGE_TYPE=local
ENV STORAGE_PATH=./data/blob_export

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

# Run application
CMD ["python", "cloud_viewer/app.py"]
