# Cross-Agent Integration Modules

This directory contains production-ready integration modules for connecting the Medical Diagnosis Agent with other agents in the fleet.

## Modules

### benchmark_integration.py

AI Benchmark Agent integration for LLM performance tracking.

**Features:**
- Real-time LLM call metrics (latency, cost, tokens)
- Per-session performance summaries
- Provider-level aggregation (Claude, OpenAI, Ollama)
- Prometheus metrics export
- Batched metric pushing to external benchmark agent

**Usage:**
```python
from agent.integrations.benchmark_integration import BenchmarkIntegration, BenchmarkContext

benchmark = BenchmarkIntegration(
    enabled=True,
    push_endpoint="http://ai-benchmark-agent:8000",
    batch_size=100,
)

# Record metrics with context manager
with BenchmarkContext(benchmark, session_id, provider, model, operation) as ctx:
    result = llm_client.complete(prompt)
    ctx.set_result(prompt_tokens=100, completion_tokens=200, cost_usd=0.01)

# Get session summary
summary = benchmark.get_session_summary(session_id)

# Export Prometheus metrics
metrics = benchmark.get_prometheus_metrics()
```

### academic_integration.py

Academic Research Agent integration for paper discovery and sharing.

**Features:**
- Submit papers to academic research knowledge base
- Query papers by topic/domain
- Automatic deduplication
- Callback registration for paper discovery
- Paper feed export

**Usage:**
```python
from agent.integrations.academic_integration import AcademicIntegration

academic = AcademicIntegration(
    enabled=True,
    push_endpoint="http://academic-research-enhanced:8000",
)

# Submit a paper
academic.submit_paper(
    title="Machine Learning for Clinical Decision Support",
    authors="Smith J, Doe A",
    year="2024",
    venue="Nature Medicine",
    url="https://doi.org/10.xxx",
    abstract="...",
    key_finding="ML models achieve 92% accuracy...",
    relevance_tags=["clinical AI", "diagnosis", "machine learning"],
)

# Query papers
papers = academic.query_papers("cardiovascular diagnosis", domain="medical")
```

### prometheus_integration.py

Prometheus metrics integration for dockprom-enhanced.

**Features:**
- Comprehensive metrics for all agent operations
- PHI-aware label scrubbing
- Histogram support for latency distributions
- Scrape endpoint handler
- Metric metadata management

**Metrics Exported:**
- `medical_diagnosis_sessions_total` — Total sessions
- `medical_triage_severity_counts` — Triage by severity
- `medical_escalations_total` — Emergency escalations
- `medical_llm_latency_seconds` — LLM request latency
- `medical_llm_cost_usd_total` — Cumulative cost
- `medical_safety_gate_triggers_total` — Safety activations
- `medical_phi_scrub_operations_total` — PHI scrub count

**Usage:**
```python
from agent.integrations.prometheus_integration import MedicalPrometheusExporter

exporter = MedicalPrometheusExporter()

# Record events
exporter.record_session_start(session_id)
exporter.record_llm_call("claude", "claude-sonnet-4-6", latency=0.5, cost=0.01, ...)
exporter.record_safety_gate_trigger("medication_dosage")

# Export for Prometheus scrape
metrics_text = exporter.export_metrics(scrub_phi=True)
```

## Security Modules

### security/production_hardening.py

Production security middleware for compliance and safety.

**Components:**

**PHIScrubber** — Removes protected health information from text:
- Names, DOBs, phone numbers, emails
- Addresses, SSNs, medical record numbers
- Configurable pattern matching
- PHI density detection

**RateLimiter** — Token-bucket rate limiting:
- Per-endpoint rules
- IP-based and user-based keys
- Burst handling with blocking
- Production-ready defaults

**AuditEncryption** — HMAC-signed audit trails:
- Tamper-evident logging
- Chain-of-hash verification
- Secure audit trail export

**SecurityMiddleware** — Unified security orchestration:
- Coordinates all security components
- Sanitizes responses and logs
- Manages security events
- Production header generation

**ComplianceLogger** — HIPAA-style compliance logging:
- Encrypted audit entries
- Session lifecycle tracking
- Security event logging
- Compliance export

**Usage:**
```python
from agent.security.production_hardening import (
    SecurityMiddleware,
    ComplianceLogger,
)

security = SecurityMiddleware()
compliance = ComplianceLogger(security)

# Check rate limit
allowed, info = security.check_rate_limit(ip_address, "api_diagnose")

# Sanitize output
clean_response = security.sanitize_response(response_data)

# Log compliance events
compliance.log_session_start(session_id, ip_hash)
compliance.log_diagnosis_result(session_id, "HIGH", True)

# Export audit trail
audit_trail = compliance.export_audit_for_compliance(hours=24)
```

## Configuration

Integration settings in `config/agent_config.yaml`:

```yaml
integrations:
  benchmark:
    enabled: true
    push_endpoint: "http://ai-benchmark-agent:8000"
    batch_size: 100

  academic:
    enabled: true
    push_endpoint: "http://academic-research-enhanced:8000"
    query_interval_hours: 168

  prometheus:
    enabled: true
    scrub_phi: true
    metrics_retention_hours: 24

security:
  rate_limiting:
    enabled: true
    default_rules: true

  phi_scrubbing:
    enabled: true
    preserve_gender: true

  audit_encryption:
    enabled: true
    key_rotation_days: 90
```

## Testing

Run integration tests:

```bash
pytest tests/test_integrations.py -v
```

## Deployment

All integrations are production-ready and automatically initialized when the agent starts. Configure via environment variables or `config/agent_config.yaml`.
