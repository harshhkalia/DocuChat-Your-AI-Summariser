FROM python:3.11-slim

# Disable telemetry and configure cache
ENV HAYSTACK_TELEMETRY_ENABLED="False"
ENV TRANSFORMERS_CACHE="/app/cache"
ENV HUGGINGFACE_HUB_CACHE="/app/cache"

# Create writable cache directory
RUN mkdir /app && chmod 777 /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip cache purge && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py pipelines.py ingestion.py ocr.py ./

# Set port and run command
ENV PORT=7860
CMD ["sh", "-c", "mkdir -p /app/cache && chmod 777 /app/cache && uvicorn app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 600"]