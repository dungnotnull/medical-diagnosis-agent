"""Prometheus metrics integration for dockprom-enhanced."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    help_text: str = ""
    metric_type: str = "gauge"
    timestamp: float = field(default_factory=time.time)


class PrometheusRegistry:
    def __init__(self):
        self._metrics: dict[str, list[MetricPoint]] = defaultdict(list)
        self._lock = threading.Lock()
        self._metric_metadata: dict[str, dict] = {}

    def register_metric(
        self,
        name: str,
        help_text: str,
        metric_type: str = "gauge",
    ) -> None:
        with self._lock:
            self._metric_metadata[name] = {
                "help": help_text,
                "type": metric_type,
            }

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
        help_text: str = "",
        metric_type: str = "gauge",
    ) -> None:
        point = MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            help_text=help_text,
            metric_type=metric_type,
        )
        with self._lock:
            self._metrics[name].append(point)
            if len(self._metrics[name]) > 1000:
                self._metrics[name] = self._metrics[name][-500:]

    def gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        self.record(name, value, labels, metric_type="gauge")

    def counter(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        self.record(name, value, labels, metric_type="counter")

    def histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[list[float]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        labels_str = self._format_labels(labels or {})
        for bucket in buckets:
            bucket_name = f"{name}_bucket"
            bucket_value = 1.0 if value <= bucket else 0.0
            bucket_labels = {**(labels or {}), "le": str(bucket)}
            self.record(bucket_name, bucket_value, bucket_labels, metric_type="histogram")
        self.record(f"{name}_bucket", 1.0, {**(labels or {}), "le": "+Inf"}, metric_type="histogram")
        self.record(f"{name}_sum", value, labels, metric_type="histogram")
        self.record(f"{name}_count", 1.0, labels, metric_type="histogram")

    def _format_labels(self, labels: dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(pairs) + "}"

    def export(self, scrub_phi: bool = True) -> str:
        lines = []
        with self._lock:
            metric_names = sorted(self._metrics.keys())

        for name in metric_names:
            metadata = self._metric_metadata.get(name, {})
            help_text = metadata.get("help", "")
            metric_type = metadata.get("type", "gauge")

            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")

            with self._lock:
                points = self._metrics.get(name, [])

            seen_label_sets = set()
            for point in points:
                if scrub_phi:
                    safe_labels = self._scrub_labels(point.labels)
                else:
                    safe_labels = point.labels
                label_signature = tuple(sorted(safe_labels.items()))
                if label_signature in seen_label_sets:
                    continue
                seen_label_sets.add(label_signature)
                labels_str = self._format_labels(safe_labels)
                lines.append(f"{name}{labels_str} {point.value}")

        return "\n".join(lines)

    def _scrub_labels(self, labels: dict[str, str]) -> dict[str, str]:
        safe = {}
        for key, value in labels.items():
            if key in ("session_id", "user_id", "patient_id"):
                import hashlib
                safe[key] = hashlib.sha256(value.encode()).hexdigest()[:16]
            elif key in ("ip_address", "email", "phone"):
                safe[key] = "[REDACTED]"
            else:
                safe[key] = value[:100]
        return safe

    def clear_old_metrics(self, age_seconds: int = 3600) -> None:
        cutoff = time.time() - age_seconds
        with self._lock:
            for name in list(self._metrics.keys()):
                self._metrics[name] = [
                    p for p in self._metrics[name] if p.timestamp >= cutoff
                ]


class MedicalPrometheusExporter:
    def __init__(self, registry: Optional[PrometheusRegistry] = None):
        self._registry = registry or PrometheusRegistry()
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        metrics = [
            ("medical_diagnosis_sessions_total", "Total number of diagnosis sessions", "counter"),
            ("medical_diagnosis_sessions_active", "Currently active diagnosis sessions", "gauge"),
            ("medical_triage_severity_counts", "Count of triage results by severity level", "gauge"),
            ("medical_escalations_total", "Total emergency escalations triggered", "counter"),
            ("medical_differential_diagnoses_returned", "Number of differential diagnoses returned per session", "gauge"),
            ("medical_safety_gate_triggers_total", "Total safety gate activations", "counter"),
            ("medical_llm_requests_total", "Total LLM API requests made", "counter"),
            ("medical_llm_latency_seconds", "LLM request latency in seconds", "gauge"),
            ("medical_llm_cost_usd_total", "Total LLM API cost in USD", "counter"),
            ("medical_llm_tokens_total", "Total LLM tokens processed", "counter"),
            ("medical_hf_model_inference_total", "Total HuggingFace model inferences", "counter"),
            ("medical_knowledge_entries_total", "Total entries in knowledge brain", "gauge"),
            ("medical_knowledge_update_duration_seconds", "Knowledge update run duration", "gauge"),
            ("medical_audit_log_entries_total", "Total encrypted audit log entries", "counter"),
            ("medical_rate_limit_denials_total", "Total rate limit denials", "counter"),
            ("medical_phi_scrub_operations_total", "Total PHI scrubbing operations", "counter"),
            ("medical_session_duration_seconds", "Duration of diagnosis sessions", "histogram"),
            ("medical_report_word_count", "Word count of generated reports", "gauge"),
            ("medical_red_flags_detected_total", "Total red flags detected", "counter"),
        ]
        for name, help_text, metric_type in metrics:
            self._registry.register_metric(name, help_text, metric_type)

    def record_session_start(self, session_id: str) -> None:
        self._registry.counter(
            "medical_diagnosis_sessions_total",
            1.0,
            labels={"agent": "medical-diagnosis"},
        )
        self._registry.gauge(
            "medical_diagnosis_sessions_active",
            1.0,
            labels={"agent": "medical-diagnosis"},
        )

    def record_session_complete(
        self,
        session_id: str,
        severity: str,
        escalation: bool,
        differential_count: int,
        duration_seconds: float,
        report_words: int,
        red_flags: int,
    ) -> None:
        self._registry.gauge(
            "medical_diagnosis_sessions_active",
            -1.0,
            labels={"agent": "medical-diagnosis"},
        )
        self._registry.gauge(
            "medical_triage_severity_counts",
            1.0,
            labels={"severity": severity, "agent": "medical-diagnosis"},
        )
        if escalation:
            self._registry.counter(
                "medical_escalations_total",
                1.0,
                labels={"severity": severity, "agent": "medical-diagnosis"},
            )
        self._registry.gauge(
            "medical_differential_diagnoses_returned",
            float(differential_count),
            labels={"agent": "medical-diagnosis"},
        )
        self._registry.histogram(
            "medical_session_duration_seconds",
            duration_seconds,
            labels={"agent": "medical-diagnosis"},
        )
        self._registry.gauge(
            "medical_report_word_count",
            float(report_words),
            labels={"agent": "medical-diagnosis"},
        )
        self._registry.counter(
            "medical_red_flags_detected_total",
            float(red_flags),
            labels={"agent": "medical-diagnosis"},
        )

    def record_llm_call(
        self,
        provider: str,
        model: str,
        latency_seconds: float,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self._registry.counter(
            "medical_llm_requests_total",
            1.0,
            labels={"provider": provider, "model": model, "agent": "medical-diagnosis"},
        )
        self._registry.gauge(
            "medical_llm_latency_seconds",
            latency_seconds,
            labels={"provider": provider, "model": model, "agent": "medical-diagnosis"},
        )
        self._registry.counter(
            "medical_llm_cost_usd_total",
            cost_usd,
            labels={"provider": provider, "agent": "medical-diagnosis"},
        )
        self._registry.counter(
            "medical_llm_tokens_total",
            float(prompt_tokens + completion_tokens),
            labels={"provider": provider, "agent": "medical-diagnosis"},
        )

    def record_hf_inference(self, model_key: str, duration_seconds: float) -> None:
        self._registry.counter(
            "medical_hf_model_inference_total",
            1.0,
            labels={"model": model_key, "agent": "medical-diagnosis"},
        )

    def record_knowledge_update(self, duration_seconds: float, new_entries: int) -> None:
        self._registry.gauge(
            "medical_knowledge_update_duration_seconds",
            duration_seconds,
            labels={"agent": "medical-diagnosis"},
        )
        self._registry.gauge(
            "medical_knowledge_entries_total",
            float(new_entries),
            labels={"agent": "medical-diagnosis"},
        )

    def record_safety_gate_trigger(self, gate_type: str) -> None:
        self._registry.counter(
            "medical_safety_gate_triggers_total",
            1.0,
            labels={"gate_type": gate_type, "agent": "medical-diagnosis"},
        )

    def record_rate_limit_denial(self, client_type: str, endpoint: str) -> None:
        self._registry.counter(
            "medical_rate_limit_denials_total",
            1.0,
            labels={"client_type": client_type, "endpoint": endpoint, "agent": "medical-diagnosis"},
        )

    def record_phi_scrub(self, text_length: int, phi_found: int) -> None:
        self._registry.counter(
            "medical_phi_scrub_operations_total",
            1.0,
            labels={"agent": "medical-diagnosis"},
        )

    def record_audit_log_entry(self, event_type: str) -> None:
        self._registry.counter(
            "medical_audit_log_entries_total",
            1.0,
            labels={"event_type": event_type, "agent": "medical-diagnosis"},
        )

    def export_metrics(self, scrub_phi: bool = True) -> str:
        return self._registry.export(scrub_phi=scrub_phi)

    def get_registry(self) -> PrometheusRegistry:
        return self._registry


class PrometheusScrapeTarget:
    def __init__(
        self,
        exporter: MedicalPrometheusExporter,
        endpoint: str = "/metrics",
        port: int = 9090,
    ):
        self._exporter = exporter
        self._endpoint = endpoint
        self._port = port
        self._scrape_count = 0
        self._last_scrape_time = None

    def handle_scrape(self, scrub_phi: bool = True) -> tuple[str, dict]:
        self._scrape_count += 1
        self._last_scrape_time = datetime.now(timezone.utc).isoformat()
        metrics_text = self._exporter.export_metrics(scrub_phi=scrub_phi)
        headers = {
            "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
            "X-Prometheus-Scrape-Count": str(self._scrape_count),
            "X-Last-Scrape-Time": self._last_scrape_time or "",
        }
        return metrics_text, headers

    def get_scrape_stats(self) -> dict:
        return {
            "endpoint": self._endpoint,
            "port": self._port,
            "total_scrapes": self._scrape_count,
            "last_scrape_time": self._last_scrape_time,
        }
