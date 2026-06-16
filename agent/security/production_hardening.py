"""Production hardening: rate limiting, PHI scrubbing, audit encryption."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# PHI patterns for scrubbing
PHI_PATTERNS = {
    "name": [
        r"\b(?:Mr|Mrs|Ms|Dr|Prof)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?=\s+is\s+a)",
    ],
    "dob": [
        r"\b(?:DOB|Date of Birth|born\s+on)[:\s]* (\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b(?=\s+(?:years?\s+old|y\.?o\.?))",
    ],
    "phone": [
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    ],
    "email": [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    ],
    "address": [
        r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b",
        r"\b(?:Apt|Apartment|Suite|Ste|#)\s+\d+\b",
    ],
    "ssn": [
        r"\b\d{3}[-]\d{2}[-]\d{4}\b",
        r"\b\d{3}\s\d{2}\s\d{4}\b",
    ],
    "mrn": [
        r"\b(?:MRN|Medical Record Number|Patient ID)[:\s]*[#]?\s*[A-Z0-9-]+\b",
    ],
}

MEDICAL_IDENTIFIERS = [
    "patient", "client", "subject", "person", "individual",
    "male", "female", "man", "woman", "boy", "girl", "child",
    "infant", "toddler", "teenager", "adult", "elderly",
]


@dataclass
class RateLimitRule:
    key: str
    requests_per_second: float
    burst: int
    block_duration_seconds: int = 300


@dataclass
class RateLimitState:
    tokens: float
    last_update: float
    blocked_until: float = 0.0


class RateLimiter:
    def __init__(self, default_rules: bool = True):
        self._rules: dict[str, RateLimitRule] = {}
        self._states: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=0.0, last_update=time.time())
        )
        self._lock = threading.Lock()
        if default_rules:
            self._apply_default_rules()

    def _apply_default_rules(self) -> None:
        self.add_rule(
            RateLimitRule(
                key="api_diagnose",
                requests_per_second=10.0,
                burst=20,
                block_duration_seconds=60,
            )
        )
        self.add_rule(
            RateLimitRule(
                key="api_knowledge_update",
                requests_per_second=0.1,
                burst=1,
                block_duration_seconds=300,
            )
        )
        self.add_rule(
            RateLimitRule(
                key="api_stats",
                requests_per_second=100.0,
                burst=200,
                block_duration_seconds=30,
            )
        )
        self.add_rule(
            RateLimitRule(
                key="global_ip",
                requests_per_second=20.0,
                burst=50,
                block_duration_seconds=120,
            )
        )

    def add_rule(self, rule: RateLimitRule) -> None:
        self._rules[rule.key] = rule

    def check_limit(self, key: str, rule_key: str = "api_diagnose") -> tuple[bool, dict]:
        rule = self._rules.get(rule_key)
        if not rule:
            return True, {"allowed": True, "reason": "No rule configured"}

        with self._lock:
            state = self._states[key]
            now = time.time()
            elapsed = now - state.last_update
            state.last_update = now

            if state.blocked_until > now:
                return False, {
                    "allowed": False,
                    "reason": "Blocked",
                    "retry_after": int(state.blocked_until - now),
                    "rule_key": rule_key,
                }

            state.tokens = min(
                rule.burst, state.tokens + elapsed * rule.requests_per_second
            )

            if state.tokens >= 1.0:
                state.tokens -= 1.0
                return True, {
                    "allowed": True,
                    "remaining_tokens": int(state.tokens),
                    "rule_key": rule_key,
                }
            else:
                state.blocked_until = now + rule.block_duration_seconds
                return False, {
                    "allowed": False,
                    "reason": "Rate limit exceeded",
                    "retry_after": rule.block_duration_seconds,
                    "rule_key": rule_key,
                }

    def get_client_key(
        self,
        ip_address: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        parts = [f"ip:{ip_address}"]
        if user_id:
            parts.append(f"user:{user_id}")
        if api_key:
            api_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            parts.append(f"key:{api_hash}")
        return ":".join(parts)

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._states:
                del self._states[key]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "rules_count": len(self._rules),
                "active_states": len([s for s in self._states.values() if s.tokens > 0]),
                "blocked_states": len(
                    [s for s in self._states.values() if s.blocked_until > time.time()]
                ),
            }


class PHIScrubber:
    def __init__(self, preserve_gender: bool = True):
        self._preserve_gender = preserve_gender
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self._compiled = {}
        for category, patterns in PHI_PATTERNS.items():
            self._compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def scrub_text(self, text: str, context: str = "medical") -> str:
        scrubbed = text
        replacements = []

        for category, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(scrubbed):
                    original = match.group()
                    placeholder = f"[{category.upper()}]"
                    if placeholder not in replacements:
                        replacements.append((original, placeholder))

        for original, placeholder in replacements:
            scrubbed = scrubbed.replace(original, placeholder, 1)

        return scrubbed

    def scrub_dict(self, data: dict, exclude_keys: Optional[set[str]] = None) -> dict:
        exclude_keys = exclude_keys or {"session_id", "created_at", "timestamp"}
        scrubbed = {}
        for key, value in data.items():
            if key in exclude_keys:
                scrubbed[key] = value
            elif isinstance(value, str):
                scrubbed[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value, exclude_keys)
            elif isinstance(value, list):
                scrubbed[key] = [
                    self.scrub_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                scrubbed[key] = value
        return scrubbed

    def scrub_log_message(self, message: str) -> str:
        return self.scrub_text(message, context="log")

    def anonymize_session_id(self, session_id: str) -> str:
        hash_bytes = hashlib.sha256(session_id.encode()).digest()
        return base64.urlsafe_b64encode(hash_bytes)[:16].decode()

    def detect_phi_density(self, text: str) -> float:
        match_count = 0
        for patterns in self._compiled.values():
            for pattern in patterns:
                match_count += len(pattern.findall(text))
        word_count = len(text.split())
        return match_count / max(word_count, 1)


class AuditEncryption:
    def __init__(self, encryption_key: Optional[str] = None):
        self._encryption_key = encryption_key or self._generate_key()
        self._algorithm = "sha256"

    def _generate_key(self) -> str:
        return base64.b64encode(hashlib.sha256(time.time().to_bytes(8, byteorder="big")).digest()).decode()

    def encrypt_audit_entry(self, entry: dict) -> str:
        entry_str = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        signature = hmac.new(
            self._encryption_key.encode(),
            entry_str.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{signature}:{base64.b64encode(entry_str.encode()).decode()}"

    def verify_audit_entry(self, encrypted: str) -> tuple[bool, Optional[dict]]:
        try:
            parts = encrypted.split(":", 1)
            if len(parts) != 2:
                return False, None
            signature, b64_data = parts
            entry_str = base64.b64decode(b64_data.encode()).decode()
            expected_signature = hmac.new(
                self._encryption_key.encode(),
                entry_str.encode(),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                return False, None
            return True, json.loads(entry_str)
        except Exception as e:
            logger.warning("Audit verification failed: %s", e)
            return False, None

    def hash_for_logging(self, value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def create_audit_chain_hash(self, previous_hash: str, entry_data: dict) -> str:
        entry_str = json.dumps(entry_data, sort_keys=True)
        combined = f"{previous_hash}:{entry_str}".encode()
        return hashlib.sha256(combined).hexdigest()


class SecurityMiddleware:
    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        phi_scrubber: Optional[PHIScrubber] = None,
        audit_encryption: Optional[AuditEncryption] = None,
    ):
        self._rate_limiter = rate_limiter or RateLimiter()
        self._phi_scrubber = phi_scrubber or PHIScrubber()
        self._audit_encryption = audit_encryption or AuditEncryption()
        self._security_events: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    def check_rate_limit(
        self,
        ip_address: str,
        endpoint: str = "api_diagnose",
        user_id: Optional[str] = None,
    ) -> tuple[bool, dict]:
        client_key = self._rate_limiter.get_client_key(ip_address, user_id)
        allowed, info = self._rate_limiter.check_limit(client_key, endpoint)
        self._log_security_event(
            "rate_limit_check",
            {"allowed": allowed, "client_hash": self._phi_scrubber.hash_for_logging(client_key), "endpoint": endpoint}
        )
        return allowed, info

    def sanitize_response(self, response_data: dict) -> dict:
        sanitized = self._phi_scrubber.scrub_dict(response_data)
        return sanitized

    def sanitize_log(self, log_message: str) -> str:
        return self._phi_scrubber.scrub_log_message(log_message)

    def encrypt_audit_log(self, log_entry: dict) -> str:
        return self._audit_encryption.encrypt_audit_entry(log_entry)

    def verify_audit_log(self, encrypted_log: str) -> tuple[bool, Optional[dict]]:
        return self._audit_encryption.verify_audit_entry(encrypted_log)

    def _log_security_event(self, event_type: str, metadata: dict) -> None:
        with self._lock:
            self._security_events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "metadata": metadata,
            })

    def get_security_summary(self) -> dict:
        with self._lock:
            events_by_type = defaultdict(int)
            for event in self._security_events:
                events_by_type[event["event_type"]] += 1
        return {
            "security_events_total": len(self._security_events),
            "events_by_type": dict(events_by_type),
            "rate_limiter_stats": self._rate_limiter.get_stats(),
        }

    def create_production_headers(self, request_context: dict) -> dict:
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }


class ComplianceLogger:
    def __init__(self, security_middleware: SecurityMiddleware):
        self._security = security_middleware
        self._audit_log: list[str] = []
        self._lock = threading.Lock()

    def log_session_start(self, session_id: str, ip_hash: str, user_id: Optional[str] = None) -> None:
        entry = {
            "event": "session_start",
            "session_id": session_id,
            "ip_hash": ip_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        encrypted = self._security.encrypt_audit_log(entry)
        with self._lock:
            self._audit_log.append(encrypted)

    def log_diagnosis_result(
        self,
        session_id: str,
        triage_severity: str,
        escalation_required: bool,
    ) -> None:
        entry = {
            "event": "diagnosis_result",
            "session_id": session_id,
            "triage_severity": triage_severity,
            "escalation_required": escalation_required,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        encrypted = self._security.encrypt_audit_log(entry)
        with self._lock:
            self._audit_log.append(encrypted)

    def log_security_event(self, event_type: str, details: dict) -> None:
        entry = {
            "event": f"security_{event_type}",
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        encrypted = self._security.encrypt_audit_log(entry)
        with self._lock:
            self._audit_log.append(encrypted)

    def get_audit_trail(self, limit: int = 100) -> list[dict]:
        with self._lock:
            recent = self._audit_log[-limit:]
        trail = []
        for encrypted in recent:
            verified, entry = self._security.verify_audit_log(encrypted)
            if verified:
                trail.append(entry)
        return trail

    def export_audit_for_compliance(self, hours: int = 24) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        lines = [
            "# Audit Trail Export",
            f"# Generated: {datetime.now(timezone.utc).isoformat()}",
            f"# Period: Last {hours} hours",
            "",
        ]
        for entry in self.get_audit_trail(limit=1000):
            if entry.get("timestamp", "") >= cutoff.isoformat():
                lines.append(json.dumps(entry))
        return "\n".join(lines)
