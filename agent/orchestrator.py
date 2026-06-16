"""Medical Diagnosis Agent orchestrator — 10-step clinical assessment pipeline."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class MedicalDiagnosisOrchestrator:
    def __init__(self, config: dict):
        self._config = config
        self._memory = None
        self._llm = None
        self._hf = None
        self._symptom_analyzer = None
        self._triage_engine = None
        self._diagnosis_engine = None
        self._report_generator = None
        self._scheduler = None
        self._session_count = 0
        self._escalation_count = 0

    def _get_memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            self._memory = MemoryManager(self._config.get("memory", {}).get("db_path", "data/medical_agent.db"))
        return self._memory

    def _get_llm(self):
        if self._llm is None:
            from tools.llm_client import UnifiedLLMClient
            self._llm = UnifiedLLMClient(memory_manager=self._get_memory())
        return self._llm

    def _get_hf(self):
        if self._hf is None:
            from tools.hf_model_manager import HFModelManager
            self._hf = HFModelManager(models_dir=self._config.get("models_dir", "models"))
        return self._hf

    def _get_symptom_analyzer(self):
        if self._symptom_analyzer is None:
            from agent.modules.symptom_analyzer import SymptomAnalyzer
            self._symptom_analyzer = SymptomAnalyzer(
                llm_client=self._get_llm(),
                hf_manager=self._get_hf(),
            )
        return self._symptom_analyzer

    def _get_triage_engine(self):
        if self._triage_engine is None:
            from agent.modules.triage_engine import TriageEngine
            self._triage_engine = TriageEngine()
        return self._triage_engine

    def _get_diagnosis_engine(self):
        if self._diagnosis_engine is None:
            from agent.modules.diagnosis_engine import DiagnosisEngine
            self._diagnosis_engine = DiagnosisEngine(
                llm_client=self._get_llm(),
                hf_manager=self._get_hf(),
                memory_manager=self._get_memory(),
            )
        return self._diagnosis_engine

    def _get_report_generator(self):
        if self._report_generator is None:
            from agent.modules.report_generator import ReportGenerator
            self._report_generator = ReportGenerator(
                llm_client=self._get_llm(),
                hf_manager=self._get_hf(),
            )
        return self._report_generator

    async def diagnose(self, patient_text: str, vitals: Optional[dict] = None) -> dict:
        session_id = str(uuid.uuid4())
        self._session_count += 1

        logger.info("Session %s: starting diagnosis pipeline", session_id)

        # Step 1: Symptom analysis
        analyzer = self._get_symptom_analyzer()
        profile = await asyncio.get_event_loop().run_in_executor(
            None, analyzer.analyze, patient_text
        )
        logger.info("Session %s: symptoms analyzed — body_system=%s red_flags=%d",
                    session_id, profile.body_system, len(profile.red_flag_keywords))

        # Step 2: Parse vitals if provided
        parsed_vitals = None
        if vitals:
            from agent.modules.triage_engine import VitalSigns
            parsed_vitals = VitalSigns(**{k: v for k, v in vitals.items() if hasattr(VitalSigns, k) or k in VitalSigns.__dataclass_fields__})

        # Step 3: Triage scoring
        triage_engine = self._get_triage_engine()
        triage = await asyncio.get_event_loop().run_in_executor(
            None, triage_engine.score, profile, parsed_vitals
        )
        logger.info("Session %s: triage=%s NEWS2=%d escalation=%s",
                    session_id, triage.severity.value, triage.news2_score, triage.escalation_required)

        # Step 4: SAFETY GATE — immediate escalation for CRITICAL
        from agent.modules.triage_engine import SeverityLevel
        if triage.severity == SeverityLevel.CRITICAL and triage.escalation_required:
            self._escalation_count += 1
            logger.warning("Session %s: CRITICAL triage — triggering emergency alert", session_id)

        # Step 5: Differential diagnosis
        dx_engine = self._get_diagnosis_engine()
        diagnosis = await asyncio.get_event_loop().run_in_executor(
            None, dx_engine.diagnose, profile, triage
        )

        # Step 6: Report generation (with safety gates)
        reporter = self._get_report_generator()
        report = await asyncio.get_event_loop().run_in_executor(
            None, reporter.generate, profile, triage, diagnosis, session_id
        )

        # Step 7: Persist to memory
        top_diff = diagnosis.differentials[0] if diagnosis.differentials else None
        self._get_memory().save_session(
            session_id=session_id,
            triage_severity=triage.severity.value,
            news2_score=triage.news2_score,
            qsofa_score=triage.qsofa_score,
            curb65_score=triage.curb65_score,
            gcs_total=triage.gcs_total,
            body_system=profile.body_system,
            red_flags=triage.red_flags,
            escalation_required=triage.escalation_required,
            top_icd_code=top_diff.icd_code if top_diff else "",
            top_condition_name=top_diff.condition_name if top_diff else "",
            patient_report_length=report.word_count,
            safety_compliant=report.safety_compliant,
            llm_provider_used=getattr(self._llm, '_last_provider', 'unknown') if self._llm else 'none',
            session_data=report.doctor_summary,
        )

        return {
            "session_id": session_id,
            "triage": {
                "severity": triage.severity.value,
                "news2_score": triage.news2_score,
                "qsofa_score": triage.qsofa_score,
                "curb65_score": triage.curb65_score,
                "gcs_total": triage.gcs_total,
                "red_flags": triage.red_flags,
                "escalation_required": triage.escalation_required,
                "recommended_action": triage.recommended_action,
            },
            "differentials": [
                {
                    "icd_code": d.icd_code,
                    "condition_name": d.condition_name,
                    "probability": d.probability,
                    "confidence_score": round(d.confidence_score, 3),
                    "urgent": d.urgent,
                    "evidence_citations": d.evidence_citations,
                }
                for d in diagnosis.differentials
            ],
            "patient_report": report.patient_report,
            "doctor_summary": report.doctor_summary,
            "safety_alerts": report.safety_alerts,
            "safety_compliant": report.safety_compliant,
        }

    def diagnose_sync(self, patient_text: str, vitals: Optional[dict] = None) -> dict:
        return asyncio.run(self.diagnose(patient_text, vitals))

    async def update_knowledge(self) -> dict:
        from tools.knowledge_updater import KnowledgeUpdater
        updater = KnowledgeUpdater(memory_manager=self._get_memory())
        return await updater.run_update()

    def get_cost_report(self) -> dict:
        return self._get_memory().get_cost_summary()

    def get_stats(self) -> dict:
        stats = self._get_memory().get_stats()
        stats["runtime_session_count"] = self._session_count
        stats["runtime_escalations"] = self._escalation_count
        return stats

    def get_prometheus_metrics(self) -> str:
        stats = self.get_stats()
        lines = [
            f"# HELP medical_sessions_total Total diagnosis sessions",
            f"# TYPE medical_sessions_total counter",
            f"medical_sessions_total {stats.get('total_sessions', 0)}",
            f"# HELP medical_escalations_total Sessions requiring emergency escalation",
            f"# TYPE medical_escalations_total counter",
            f"medical_escalations_total {stats.get('escalations', 0)}",
        ]
        for severity, count in stats.get("by_severity", {}).items():
            lines.append(f'medical_triage_severity{{level="{severity}"}} {count}')
        return "\n".join(lines)

    def start_scheduler(self) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                self._scheduled_knowledge_update,
                CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="weekly_knowledge_update",
            )
            self._scheduler.start()
            logger.info("APScheduler started — weekly knowledge update Sunday 02:00")
        except ImportError:
            logger.warning("APScheduler not installed — scheduled updates disabled")

    def _scheduled_knowledge_update(self) -> None:
        asyncio.run(self.update_knowledge())
