"""Academic Research Agent integration for additional paper discovery."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperSubmission:
    title: str
    authors: str
    year: str
    venue: str
    url: str
    abstract: str
    key_finding: str
    relevance_tags: list[str]
    submitted_at: str


class AcademicIntegration:
    _instance: Optional["AcademicIntegration"] = None
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
        query_interval_hours: int = 168,
    ):
        if self._initialized:
            return
        self._initialized = True
        self._enabled = enabled
        self._push_endpoint = push_endpoint
        self._query_interval_hours = query_interval_hours
        self._submitted_papers: dict[str, str] = {}
        self._lock = threading.Lock()
        logger.info("AcademicIntegration initialized (enabled=%s)", enabled)

    def submit_paper(
        self,
        title: str,
        authors: str,
        year: str,
        venue: str,
        url: str,
        abstract: str,
        key_finding: str,
        relevance_tags: list[str],
    ) -> bool:
        if not self._enabled:
            return False

        import hashlib
        paper_hash = hashlib.sha256(f"{title}{url}".encode()).hexdigest()
        if paper_hash in self._submitted_papers:
            return False

        submission = PaperSubmission(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            url=url,
            abstract=abstract,
            key_finding=key_finding,
            relevance_tags=relevance_tags,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

        success = self._push_to_academic_agent(submission)
        if success:
            with self._lock:
                self._submitted_papers[paper_hash] = submission.submitted_at
            logger.info("Submitted paper to academic agent: %s", title[:60])
        return success

    def _push_to_academic_agent(self, submission: PaperSubmission) -> bool:
        if not self._push_endpoint:
            logger.debug("No push endpoint configured — paper not submitted")
            return False

        try:
            payload = json.dumps({
                "title": submission.title,
                "authors": submission.authors,
                "year": submission.year,
                "venue": submission.venue,
                "url": submission.url,
                "abstract": submission.abstract,
                "key_finding": submission.key_finding,
                "relevance_tags": submission.relevance_tags,
                "submitted_at": submission.submitted_at,
                "source_agent": "medical-diagnosis-agent",
            }).encode()

            req = urllib.request.Request(
                f"{self._push_endpoint}/api/v1/papers/ingest",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("Failed to push paper to academic agent: %s", e)
            return False

    def query_papers(
        self,
        query: str,
        domain: str = "medical",
        max_results: int = 10,
    ) -> list[dict]:
        if not self._enabled or not self._push_endpoint:
            return []

        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"{self._push_endpoint}/api/v1/papers/search"
                f"?q={encoded_query}&domain={domain}&limit={max_results}"
            )
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read())
                return data.get("results", [])
        except Exception as e:
            logger.warning("Failed to query academic agent: %s", e)
            return []

    def get_medical_papers_by_topic(
        self,
        topic: str,
        recency_days: int = 365,
    ) -> list[dict]:
        topics_map = {
            "cardiovascular": ["cardiology", "heart disease", "cardiac"],
            "respiratory": ["pulmonology", "respiratory", "lung"],
            "neurological": ["neurology", "brain", "stroke"],
            "gastrointestinal": ["gastroenterology", "digestive", "gi"],
            "infectious": ["infectious disease", "virology", "bacteriology"],
            "emergency": ["emergency medicine", "triage", "critical care"],
        }
        search_terms = topics_map.get(topic, [topic])
        all_results = []
        for term in search_terms[:2]:
            results = self.query_papers(
                query=f"{term} clinical diagnosis guidelines",
                domain="medical",
                max_results=5,
            )
            all_results.extend(results)
        return self._dedupe_papers(all_results)[:10]

    def _dedupe_papers(self, papers: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for paper in papers:
            identifier = paper.get("url", "") or paper.get("title", "")
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique.append(paper)
        return unique

    def register_callback_urls(self, callback_urls: list[str]) -> None:
        if not self._enabled or not self._push_endpoint:
            return
        try:
            payload = json.dumps({
                "callback_urls": callback_urls,
                "agent": "medical-diagnosis-agent",
            }).encode()
            req = urllib.request.Request(
                f"{self._push_endpoint}/api/v1/agents/register",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Registered callback URLs with academic agent")
        except Exception as e:
            logger.warning("Failed to register callbacks: %s", e)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "submitted_papers_count": len(self._submitted_papers),
                "push_endpoint_configured": bool(self._push_endpoint),
                "query_interval_hours": self._query_interval_hours,
            }

    def export_paper_feed(self, limit: int = 50) -> str:
        lines = [
            "# Medical Diagnosis Agent — Paper Feed",
            f"# Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Format: | Title | Authors | Year | Venue | URL | Key Finding |",
            "",
        ]
        with self._lock:
            recent_hashes = list(self._submitted_papers.items())[-limit:]
        lines.extend([
            f"| {h} | {ts} |"
            for h, ts in recent_hashes
        ])
        return "\n".join(lines)
