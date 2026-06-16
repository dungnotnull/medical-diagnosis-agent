"""AI Benchmark Agent integration for LLM performance tracking."""
from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetric:
    session_id: str
    provider: str
    model: str
    operation: str
    start_time: float
    end_time: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    success: bool
    error_message: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def tokens_per_second(self) -> float:
        total_tokens = self.prompt_tokens + self.completion_tokens
        duration = self.end_time - self.start_time
        return total_tokens / duration if duration > 0 else 0.0


class BenchmarkIntegration:
    _instance: Optional["BenchmarkIntegration"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        enabled: bool = True,
        push_endpoint: Optional[str] = None,
        batch_size: int = 100,
    ):
        if self._initialized:
            return
        self._initialized = True
        self._enabled = enabled
        self._push_endpoint = push_endpoint
        self._batch_size = batch_size
        self._metrics: list[BenchmarkMetric] = []
        self._session_metrics: dict[str, dict] = defaultdict(lambda: {
            "total_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "call_count": 0,
            "errors": 0,
        })
        self._lock = threading.Lock()
        logger.info("BenchmarkIntegration initialized (enabled=%s)", enabled)

    def record_llm_call(
        self,
        session_id: str,
        provider: str,
        model: str,
        operation: str,
        start_time: float,
        end_time: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        success: bool,
        error_message: str = "",
    ) -> None:
        if not self._enabled:
            return

        metric = BenchmarkMetric(
            session_id=session_id,
            provider=provider,
            model=model,
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            success=success,
            error_message=error_message,
        )

        with self._lock:
            self._metrics.append(metric)
            self._session_metrics[session_id]["total_latency_ms"] += metric.latency_ms
            self._session_metrics[session_id]["total_cost_usd"] += cost_usd
            self._session_metrics[session_id]["total_tokens"] += (
                prompt_tokens + completion_tokens
            )
            self._session_metrics[session_id]["call_count"] += 1
            if not success:
                self._session_metrics[session_id]["errors"] += 1

        if len(self._metrics) >= self._batch_size:
            self._flush_metrics()

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        with self._lock:
            if session_id not in self._session_metrics:
                return None
            stats = self._session_metrics[session_id]
            return {
                "session_id": session_id,
                "total_latency_ms": round(stats["total_latency_ms"], 2),
                "total_cost_usd": round(stats["total_cost_usd"], 6),
                "total_tokens": stats["total_tokens"],
                "call_count": stats["call_count"],
                "avg_latency_ms": round(
                    stats["total_latency_ms"] / stats["call_count"]
                    if stats["call_count"] > 0
                    else 0,
                    2,
                ),
                "error_rate": round(
                    stats["errors"] / stats["call_count"] if stats["call_count"] > 0 else 0,
                    4,
                ),
            }

    def get_aggregate_stats(self, hours: int = 24) -> dict:
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            recent = [m for m in self._metrics if m.start_time >= cutoff]
        if not recent:
            return {
                "total_calls": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "success_rate": 1.0,
            }

        total_calls = len(recent)
        total_cost = sum(m.cost_usd for m in recent)
        total_latency = sum(m.latency_ms for m in recent)
        success_count = sum(1 for m in recent if m.success)

        return {
            "period_hours": hours,
            "total_calls": total_calls,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(total_latency / total_calls, 2),
            "success_rate": round(success_count / total_calls, 4),
            "total_tokens": sum(m.prompt_tokens + m.completion_tokens for m in recent),
            "by_provider": self._aggregate_by_provider(recent),
            "by_model": self._aggregate_by_model(recent),
        }

    def _aggregate_by_provider(self, metrics: list[BenchmarkMetric]) -> dict:
        provider_stats = defaultdict(lambda: {"calls": 0, "cost": 0.0, "latency": 0.0})
        for m in metrics:
            provider_stats[m.provider]["calls"] += 1
            provider_stats[m.provider]["cost"] += m.cost_usd
            provider_stats[m.provider]["latency"] += m.latency_ms
        return {
            provider: {
                "calls": stats["calls"],
                "total_cost_usd": round(stats["cost"], 4),
                "avg_latency_ms": round(stats["latency"] / stats["calls"], 2),
            }
            for provider, stats in provider_stats.items()
        }

    def _aggregate_by_model(self, metrics: list[BenchmarkMetric]) -> dict:
        model_stats = defaultdict(lambda: {"calls": 0, "cost": 0.0, "latency": 0.0})
        for m in metrics:
            model_stats[m.model]["calls"] += 1
            model_stats[m.model]["cost"] += m.cost_usd
            model_stats[m.model]["latency"] += m.latency_ms
        return {
            model: {
                "calls": stats["calls"],
                "total_cost_usd": round(stats["cost"], 4),
                "avg_latency_ms": round(stats["latency"] / stats["calls"], 2) if stats["calls"] > 0 else 0,
            }
            for model, stats in model_stats.items()
        }

    def get_prometheus_metrics(self) -> str:
        stats = self.get_aggregate_stats(hours=1)
        lines = [
            "# HELP medical_benchmark_llm_calls_total Total LLM calls",
            "# TYPE medical_benchmark_llm_calls_total gauge",
            f"medical_benchmark_llm_calls_total {stats['total_calls']}",
            "# HELP medical_benchmark_llm_cost_usd_total Total LLM cost in USD",
            "# TYPE medical_benchmark_llm_cost_usd_total gauge",
            f"medical_benchmark_llm_cost_usd_total {stats['total_cost_usd']}",
            "# HELP medical_benchmark_llm_avg_latency_ms Average LLM latency",
            "# TYPE medical_benchmark_llm_avg_latency_ms gauge",
            f"medical_benchmark_llm_avg_latency_ms {stats['avg_latency_ms']}",
            "# HELP medical_benchmark_llm_success_rate LLM call success rate",
            "# TYPE medical_benchmark_llm_success_rate gauge",
            f"medical_benchmark_llm_success_rate {stats['success_rate']}",
        ]
        for provider, metrics in stats.get("by_provider", {}).items():
            lines.extend([
                f'medical_benchmark_provider_calls{{provider="{provider}"}} {metrics["calls"]}',
                f'medical_benchmark_provider_cost_usd{{provider="{provider}"}} {metrics["total_cost_usd"]}',
                f'medical_benchmark_provider_latency_ms{{provider="{provider}"}} {metrics["avg_latency_ms"]}',
            ])
        for model, metrics in stats.get("by_model", {}).items():
            lines.extend([
                f'medical_benchmark_model_calls{{model="{model}"}} {metrics["calls"]}',
                f'medical_benchmark_model_cost_usd{{model="{model}"}} {metrics["total_cost_usd"]}',
            ])
        return "\n".join(lines)

    def _flush_metrics(self) -> None:
        if not self._push_endpoint:
            return
        import json
        import urllib.request
        try:
            payload = json.dumps([{
                "session_id": m.session_id,
                "provider": m.provider,
                "model": m.model,
                "operation": m.operation,
                "latency_ms": m.latency_ms,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "cost_usd": m.cost_usd,
                "success": m.success,
                "timestamp": datetime.fromtimestamp(m.start_time, tz=timezone.utc).isoformat(),
            } for m in self._metrics[:self._batch_size]]).encode()
            req = urllib.request.Request(
                f"{self._push_endpoint}/api/v1/benchmark/ingest",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    with self._lock:
                        self._metrics = self._metrics[self._batch_size:]
                    logger.debug("Flushed %d metrics to benchmark agent", self._batch_size)
        except Exception as e:
            logger.warning("Failed to flush metrics to benchmark agent: %s", e)


class BenchmarkContext:
    def __init__(
        self,
        integration: BenchmarkIntegration,
        session_id: str,
        provider: str,
        model: str,
        operation: str,
    ):
        self._integration = integration
        self._session_id = session_id
        self._provider = provider
        self._model = model
        self._operation = operation
        self._start_time = 0.0
        self._end_time = 0.0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0
        self._success = True
        self._error_message = ""

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        if exc_type is not None:
            self._success = False
            self._error_message = str(exc_val) if exc_val else "Unknown error"
        self._integration.record_llm_call(
            session_id=self._session_id,
            provider=self._provider,
            model=self._model,
            operation=self._operation,
            start_time=self._start_time,
            end_time=self._end_time,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            cost_usd=self._cost_usd,
            success=self._success,
            error_message=self._error_message,
        )
        return False

    def set_result(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost_usd = cost_usd
