FROM python:3.12-slim

LABEL maintainer="medical-diagnosis-agent"
LABEL description="Evidence-based clinical triage and differential diagnosis AI agent"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash agentuser
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=agentuser:agentuser . .

# Create data directories
RUN mkdir -p data models && chown -R agentuser:agentuser /app

USER agentuser

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

CMD ["python", "-m", "agent.main", "serve", "--host", "0.0.0.0", "--port", "8008"]
