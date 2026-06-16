# Medical Diagnosis Agent — Production Deployment Guide

## Overview

This guide covers deploying the Medical Diagnosis Agent to production with all Phase 7 integrations enabled.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.12+ (for local development)
- Valid API keys for LLM providers (Anthropic Claude, OpenAI) or local Ollama
- (Optional) NVIDIA GPU for HuggingFace model acceleration

## Environment Configuration

Copy `config/.env.example` to `config/.env` and configure:

```bash
# Required LLM API keys
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx

# Privacy mode (set to true for PHI handling)
PRIVACY_MODE=false

# Optional: HuggingFace token for gated models
HF_TOKEN=hf_xxxxx

# Optional: NCBI API for higher rate limits
NCBI_EMAIL=your_email@example.com
```

## Deployment Options

### 1. Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Start with Ollama for offline mode
docker-compose --profile ollama up -d

# Check health
curl http://localhost:8008/health
```

### 2. Docker with GPU Support

```bash
# Build GPU-enabled image
docker-compose --profile gpu up -d

# Verify GPU access
docker exec medical-diagnosis-agent python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Run server
python -m agent.main serve --host 0.0.0.0 --port 8008 --start-scheduler
```

## Cross-Agent Integration

### AI Benchmark Agent Integration

The agent automatically records LLM performance metrics when configured:

```python
from agent.integrations.benchmark_integration import BenchmarkIntegration

benchmark = BenchmarkIntegration(
    enabled=True,
    push_endpoint="http://ai-benchmark-agent:8000",
    batch_size=100,
)
```

Metrics tracked:
- Per-session latency, cost, token usage
- Aggregate provider stats (Claude, OpenAI, Ollama)
- Success/error rates

### Academic Research Agent Integration

Enable automatic paper discovery and sharing:

```python
from agent.integrations.academic_integration import AcademicIntegration

academic = AcademicIntegration(
    enabled=True,
    push_endpoint="http://academic-research-enhanced:8000",
    query_interval_hours=168,  # Weekly
)
```

### Prometheus Metrics

Metrics exposed at `http://localhost:8008/metrics`:

```bash
# Scrape metrics
curl http://localhost:8008/metrics

# Key metrics:
# - medical_diagnosis_sessions_total
# - medical_triage_severity_counts
# - medical_escalations_total
# - medical_llm_cost_usd_total
# - medical_safety_gate_triggers_total
```

## Security & Compliance

### PHI Scrubbing

All logs and audit trails automatically scrub:

```python
from agent.security.production_hardening import PHIScrubber

scrubber = PHIScrubber()
clean_text = scrubber.scrub_log_message("Patient John Doe reported chest pain")
# Returns: "Patient [NAME] reported chest pain"
```

### Rate Limiting

Default rate limits (configurable in `config/agent_config.yaml`):

```yaml
rate_limiting:
  api_diagnose:
    requests_per_second: 10
    burst: 20
    block_duration_seconds: 60
```

### Audit Trail Encryption

All audit logs are HMAC-signed for tamper evidence:

```python
from agent.security.production_hardening import AuditEncryption

audit = AuditEncryption(encryption_key="your-secret-key")
encrypted_entry = audit.encrypt_audit_entry({"session_id": "xxx", "event": "diagnosis"})
```

## Health Checks

```bash
# Basic health
curl http://localhost:8008/health

# With metrics
curl http://localhost:8008/metrics

# Run diagnosis test
curl -X POST http://localhost:8008/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "mild headache for 2 days"}'
```

## Monitoring

### Prometheus Scrape Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'medical-diagnosis-agent'
    static_configs:
      - targets: ['medical-diagnosis-agent:8008']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics to Monitor

- `medical_diagnosis_sessions_total` — Total sessions processed
- `medical_triage_severity_counts{severity="CRITICAL"}` — Critical triages
- `medical_escalations_total` — Emergency escalations
- `medical_safety_gate_triggers_total` — Safety gate activations
- `medical_llm_cost_usd_total` — Cumulative LLM costs

## Troubleshooting

### HuggingFace Models Not Downloading

```bash
# Set HF_TOKEN in .env for gated models
HF_TOKEN=hf_xxxxx

# Or use CPU-only mode
export TRANSFORMERS_OFFLINE=1
```

### Ollama Connection Issues

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check Ollama service in Docker
docker-compose logs ollama
```

### High Memory Usage

```python
# Adjust model cache timeout in config/agent_config.yaml
hf_models:
  idle_timeout_seconds: 300  # Unload after 5 minutes
```

## Production Checklist

- [ ] All API keys configured in `.env`
- [ ] Rate limiting enabled and tested
- [ ] PHI scrubbing verified with test data
- [ ] Audit trail encryption configured
- [ ] Prometheus scrape target added
- [ ] Health check endpoint accessible
- [ ] Database persistence configured (`data/` volume mounted)
- [ ] Log rotation configured
- [ ] Backup strategy for SQLite database
- [ ] Privacy mode tested with PHI data
- [ ] Safety gates tested with medication dosage prompts

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.override.yml
services:
  medical-diagnosis-agent:
    deploy:
      replicas: 3
    environment:
      - INSTANCE_ID=${HOSTNAME}
```

### Database Scaling

For high-volume deployments, consider migrating from SQLite to PostgreSQL:

```python
# In agent/memory/memory_manager.py
# Change: sqlite3.connect() to: psycopg2.connect()
```

## Support

For issues or questions:
- GitHub: https://github.com/your-org/medical-diagnosis-agent
- Documentation: See CLAUDE.md and PROJECT-detail.md
