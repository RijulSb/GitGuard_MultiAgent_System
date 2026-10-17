FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install CPU-only PyTorch first for fast, lightweight build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY configs/ ./configs/
COPY data/ ./data/
COPY src/ ./src/
COPY start.sh .
COPY .env.example .

# Set execution permissions on start script
RUN chmod +x start.sh

# Expose ports for FastAPI (8000) and Streamlit UI (8501)
EXPOSE 8000 8501

# Healthcheck for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint launches both API and UI
ENTRYPOINT ["./start.sh"]
CMD ["all"]
