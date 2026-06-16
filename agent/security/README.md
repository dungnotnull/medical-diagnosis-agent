# Security & Compliance Modules

Production-hardened security modules for HIPAA-style compliance and PHI protection.

## Overview

These modules provide enterprise-grade security features required for production deployment in healthcare environments:

- **Rate Limiting** — Token-bucket algorithm with configurable rules
- **PHI Scrubbing** — Automatic detection and redaction of protected health information
- **Audit Encryption** — HMAC-signed audit trails for tamper evidence
- **Security Middleware** — Unified security orchestration
- **Compliance Logging** — HIPAA-style audit and compliance reporting

## Quick Start

```python
from agent.security.production_hardening import (
    SecurityMiddleware,
    ComplianceLogger,
    PHIScrubber,
    RateLimiter,
)

# Initialize security middleware
security = SecurityMiddleware()

# Initialize compliance logger
compliance = ComplianceLogger(security)

# Check rate limit before processing
allowed, info = security.check_rate_limit("192.168.1.100", "api_diagnose")
if not allowed:
    return {"error": "Rate limit exceeded", "retry_after": info["retry_after"]}

# Process request
result = orchestrator.diagnose(symptoms)

# Sanitize response
clean_result = security.sanitize_response(result)

# Log compliance events
compliance.log_session_start(session_id, ip_hash="a3f2b1c4")
compliance.log_diagnosis_result(session_id, "HIGH", True)
```

## Components

### PHIScrubber

Detects and redacts Protected Health Information (PHI) from text.

**Patterns Detected:**
- Names (Mr/Mrs/Dr + First Last, First Last)
- Dates of birth
- Phone numbers
- Email addresses
- Street addresses
- Social Security Numbers
- Medical Record Numbers (MRN)
- Patient identifiers

```python
scrubber = PHIScrubber(preserve_gender=True)

# Scrub text
clean = scrubber.scrub_text("Patient John Doe called from 555-123-4567")
# Returns: "Patient [NAME] called from [PHONE]"

# Scrub dictionary response
clean_response = scrubber.scrub_dict(response_data, exclude_keys={"session_id"})

# Detect PHI density
phi_density = scrubber.detect_phi_density(text)  # Returns: 0.05 (5% of words are PHI)
```

### RateLimiter

Token-bucket rate limiting with burst handling and automatic blocking.

**Default Rules:**
- `api_diagnose`: 10 req/s, burst 20
- `api_knowledge_update`: 0.1 req/s (1 per 10s), burst 1
- `api_stats`: 100 req/s, burst 200
- `global_ip`: 20 req/s, burst 50

```python
limiter = RateLimiter(default_rules=True)

# Add custom rule
from agent.security.production_hardening import RateLimitRule
limiter.add_rule(RateLimitRule(
    key="custom_endpoint",
    requests_per_second=5.0,
    burst=10,
    block_duration_seconds=60,
))

# Check limit
client_key = limiter.get_client_key(ip_address="192.168.1.100", user_id="user123")
allowed, info = limiter.check_limit(client_key, rule_key="api_diagnose")

# Get stats
stats = limiter.get_stats()
```

### AuditEncryption

HMAC-signed audit entries for tamper-evident logging.

```python
audit = AuditEncryption(encryption_key="your-secret-key")

# Encrypt audit entry
encrypted = audit.encrypt_audit_entry({
    "event": "session_start",
    "session_id": "abc-123",
    "timestamp": "2024-01-01T00:00:00Z",
})

# Verify and decrypt
verified, entry = audit.verify_audit_entry(encrypted)

# Create chain hash for sequential audit trail
chain_hash = audit.create_audit_chain_hash(previous_hash, new_entry)
```

### SecurityMiddleware

Unified security orchestration coordinating all components.

```python
security = SecurityMiddleware()

# Rate limit check
allowed, info = security.check_rate_limit(ip="192.168.1.100", endpoint="api_diagnose")

# Sanitize data
clean_response = security.sanitize_response(response_data)

# Sanitize logs
clean_log = security.sanitize_log("Patient John Doe reported chest pain")

# Encrypt audit logs
encrypted_entry = security.encrypt_audit_log({"event": "diagnosis", "severity": "HIGH"})

# Get security summary
summary = security.get_security_summary()

# Get production headers
headers = security.create_production_headers({"request_ip": "192.168.1.100"})
```

### ComplianceLogger

HIPAA-style compliance logging with encrypted audit trails.

```python
compliance = ComplianceLogger(security_middleware)

# Log session events
compliance.log_session_start(session_id="xyz-789", ip_hash="hashed_ip", user_id="user123")
compliance.log_diagnosis_result(session_id, triage_severity="CRITICAL", escalation_required=True)
compliance.log_security_event("rate_limit_exceeded", {"ip": "192.168.1.100"})

# Get audit trail
trail = compliance.get_audit_trail(limit=100)

# Export for compliance reporting
export = compliance.export_audit_for_compliance(hours=24)
```

## Configuration

Configure security settings in `config/agent_config.yaml`:

```yaml
security:
  enabled: true

  rate_limiting:
    enabled: true
    default_rules: true
    custom_rules: []

  phi_scrubbing:
    enabled: true
    preserve_gender: true
    additional_patterns: []

  audit_encryption:
    enabled: true
    key_rotation_days: 90

  compliance_logging:
    enabled: true
    audit_retention_days: 365
```

## Production Deployment

### Environment Variables

```bash
# Security configuration
SECURITY_ENABLED=true
RATE_LIMITING_ENABLED=true
PHI_SCRUBBING_ENABLED=true
AUDIT_ENCRYPTION_KEY=your-secret-key-here

# Compliance settings
AUDIT_RETENTION_DAYS=365
COMPLIANCE_EXPORT_ENABLED=true
```

### Docker Configuration

```yaml
services:
  medical-diagnosis-agent:
    environment:
      - SECURITY_ENABLED=true
      - AUDIT_ENCRYPTION_KEY=${AUDIT_KEY}
    volumes:
      - ./audit_logs:/app/audit
```

## Monitoring

Monitor security metrics via Prometheus:

- `medical_rate_limit_denials_total` — Rate limit blocks
- `medical_phi_scrub_operations_total` — PHI scrub count
- `medical_audit_log_entries_total` — Audit entries written

## Compliance Features

- **HIPAA-ready** audit trails with HMAC signing
- **PHI protection** with automatic detection and redaction
- **Rate limiting** to prevent abuse and ensure availability
- **Tamper evidence** through cryptographic audit chains
- **Production headers** for web security (CSP, HSTS, XSS protection)
- **Compliance export** for regulatory reporting

## Best Practices

1. **Always enable PHI scrubbing** when handling real patient data
2. **Rotate audit encryption keys** regularly (90-day default)
3. **Monitor rate limit denials** for potential abuse patterns
4. **Export audit trails** regularly for backup and compliance
5. **Use privacy mode** (Ollama) when processing sensitive data
6. **Test with PHI samples** to verify scrubbing effectiveness

## Testing

Run security tests:

```bash
pytest tests/test_security.py -v
```

Test PHI scrubbing:

```bash
python -m agent.security.production_hardening
# Interactive testing mode
```
