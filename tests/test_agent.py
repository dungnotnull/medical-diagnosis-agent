"""Automated tests for the Medical Diagnosis Agent."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── SymptomAnalyzer Tests ───────────────────────────────────────────────────

class TestSymptomAnalyzer(unittest.TestCase):
    def setUp(self):
        from agent.modules.symptom_analyzer import SymptomAnalyzer
        self.analyzer = SymptomAnalyzer(llm_client=None, hf_manager=None)

    def test_red_flag_chest_pain_detected(self):
        profile = self.analyzer.analyze("I have severe chest pain and I'm sweating a lot")
        self.assertIn("chest pain", " ".join(profile.red_flag_keywords).lower())

    def test_red_flag_stroke_fast_detected(self):
        profile = self.analyzer.analyze("My husband has facial drooping and slurred speech")
        combined = " ".join(profile.red_flag_keywords).lower()
        self.assertTrue("droop" in combined or "speech" in combined or "stroke" in combined)

    def test_body_system_cardiovascular(self):
        profile = self.analyzer.analyze("severe chest pain with palpitations")
        self.assertEqual(profile.body_system, "cardiovascular")

    def test_body_system_neurological(self):
        profile = self.analyzer.analyze("sudden severe headache and dizziness")
        self.assertEqual(profile.body_system, "neurological")

    def test_body_system_respiratory(self):
        profile = self.analyzer.analyze("cough with shortness of breath for 3 days")
        self.assertEqual(profile.body_system, "respiratory")

    def test_is_immediately_critical(self):
        profile = self.analyzer.analyze("patient is unconscious and not breathing")
        self.assertTrue(profile.is_immediately_critical())

    def test_entities_extracted(self):
        profile = self.analyzer.analyze("I have pain and fever with nausea")
        self.assertGreater(len(profile.extracted_entities), 0)

    def test_heuristic_opqrst_severity(self):
        profile = self.analyzer.analyze("severe headache 8/10")
        self.assertIn("8", str(profile.severity))

    def test_profile_valid_with_minimal_text(self):
        profile = self.analyzer.analyze("headache since 2 hours ago")
        # Should not crash; completeness may be low
        self.assertIsNotNone(profile.opqrst_completeness)

    def test_low_risk_no_red_flags(self):
        profile = self.analyzer.analyze("mild sore throat for 1 day, no fever")
        combined = " ".join(profile.red_flag_keywords).lower()
        self.assertNotIn("chest pain", combined)


# ─── TriageEngine Tests ──────────────────────────────────────────────────────

class TestTriageEngine(unittest.TestCase):
    def setUp(self):
        from agent.modules.triage_engine import TriageEngine, SeverityLevel
        from agent.modules.symptom_analyzer import SymptomAnalyzer
        self.engine = TriageEngine()
        self.analyzer = SymptomAnalyzer(llm_client=None, hf_manager=None)
        self.SeverityLevel = SeverityLevel

    def test_critical_chest_pain_diaphoresis(self):
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="crushing chest pain with sweating and diaphoresis")
        profile.red_flag_keywords = ["chest pain", "diaphoresis"]
        result = self.engine.score(profile)
        self.assertEqual(result.severity, self.SeverityLevel.CRITICAL)
        self.assertTrue(result.escalation_required)

    def test_critical_stroke_fast(self):
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="facial drooping slurred speech arm weakness sudden onset")
        result = self.engine.score(profile)
        self.assertEqual(result.severity, self.SeverityLevel.CRITICAL)

    def test_news2_calculation_with_vitals(self):
        from agent.modules.triage_engine import VitalSigns
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="patient unwell with fever")
        vitals = VitalSigns(
            respiration_rate=25,
            spo2=93.0,
            systolic_bp=95,
            pulse=115,
            temperature=38.5,
            consciousness="A",
        )
        result = self.engine.score(profile, vitals)
        self.assertGreater(result.news2_score, 0)
        self.assertFalse(result.news2_partial)

    def test_gcs_critical_threshold(self):
        from agent.modules.triage_engine import VitalSigns
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="found unresponsive")
        vitals = VitalSigns(gcs_eye=2, gcs_verbal=2, gcs_motor=4)
        result = self.engine.score(profile, vitals)
        self.assertEqual(result.gcs_total, 8)
        self.assertEqual(result.severity, self.SeverityLevel.CRITICAL)

    def test_qsofa_sepsis_flag(self):
        from agent.modules.triage_engine import VitalSigns
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="confusion and suspected infection")
        vitals = VitalSigns(
            respiration_rate=24,
            systolic_bp=98,
            consciousness="C",
        )
        result = self.engine.score(profile, vitals)
        self.assertGreaterEqual(result.qsofa_score, 2)

    def test_curb65_severe_pneumonia(self):
        from agent.modules.triage_engine import VitalSigns
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="cough fever confusion elderly patient")
        vitals = VitalSigns(
            respiration_rate=31,
            systolic_bp=88,
            age=72,
            consciousness="C",
        )
        result = self.engine.score(profile, vitals)
        self.assertGreaterEqual(result.curb65_score, 3)

    def test_low_severity_mild_symptoms(self):
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="mild sore throat and slight runny nose")
        result = self.engine.score(profile)
        self.assertEqual(result.severity, self.SeverityLevel.LOW)

    def test_recommended_action_critical(self):
        from agent.modules.symptom_analyzer import SymptomProfile
        profile = SymptomProfile(raw_text="unconscious patient not breathing")
        result = self.engine.score(profile)
        self.assertIn("911", result.recommended_action.upper() + result.recommended_action)


# ─── DiagnosisEngine Tests ───────────────────────────────────────────────────

class TestDiagnosisEngine(unittest.TestCase):
    def setUp(self):
        from agent.modules.diagnosis_engine import DiagnosisEngine
        self.engine = DiagnosisEngine(llm_client=None, hf_manager=None)

    def _make_profile(self, text: str, system: str = "cardiovascular"):
        from agent.modules.symptom_analyzer import SymptomProfile
        p = SymptomProfile(raw_text=text)
        p.body_system = system
        p.extracted_entities = ["pain", "chest"]
        return p

    def _make_triage(self, severity: str = "HIGH"):
        from agent.modules.triage_engine import TriageResult, SeverityLevel
        t = TriageResult()
        t.severity = SeverityLevel[severity]
        t.news2_score = 6
        return t

    def test_returns_differentials(self):
        profile = self._make_profile("chest pain with diaphoresis")
        triage = self._make_triage("HIGH")
        result = self.engine.diagnose(profile, triage)
        self.assertGreater(len(result.differentials), 0)

    def test_min_two_differentials(self):
        profile = self._make_profile("chest pain", "cardiovascular")
        triage = self._make_triage("MEDIUM")
        result = self.engine.diagnose(profile, triage)
        self.assertGreaterEqual(len(result.differentials), 2)

    def test_all_differentials_have_icd_codes(self):
        profile = self._make_profile("cough fever", "respiratory")
        triage = self._make_triage("LOW")
        result = self.engine.diagnose(profile, triage)
        for d in result.differentials:
            self.assertNotEqual(d.icd_code, "")

    def test_all_differentials_have_citations(self):
        profile = self._make_profile("abdominal pain nausea", "gastrointestinal")
        triage = self._make_triage("MEDIUM")
        result = self.engine.diagnose(profile, triage)
        for d in result.differentials:
            self.assertGreater(len(d.evidence_citations), 0)

    def test_confidence_score_range(self):
        profile = self._make_profile("headache dizziness", "neurological")
        triage = self._make_triage("HIGH")
        result = self.engine.diagnose(profile, triage)
        for d in result.differentials:
            self.assertGreaterEqual(d.confidence_score, 0.0)
            self.assertLessEqual(d.confidence_score, 1.0)

    def test_disclaimer_present(self):
        profile = self._make_profile("pain")
        triage = self._make_triage()
        result = self.engine.diagnose(profile, triage)
        self.assertIn("informational", result.disclaimer.lower())


# ─── ReportGenerator Tests ───────────────────────────────────────────────────

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        from agent.modules.report_generator import ReportGenerator
        self.generator = ReportGenerator(llm_client=None, hf_manager=None)

    def _build_inputs(self, severity: str = "LOW"):
        from agent.modules.symptom_analyzer import SymptomProfile
        from agent.modules.triage_engine import TriageResult, SeverityLevel
        from agent.modules.diagnosis_engine import DiagnosisResult, DiagnosisCandidate

        profile = SymptomProfile(raw_text="test patient symptoms")
        profile.body_system = "respiratory"
        profile.extracted_entities = ["cough"]

        triage = TriageResult()
        triage.severity = SeverityLevel[severity]
        triage.news2_score = 3 if severity == "MEDIUM" else 1
        triage.recommended_action = "See GP within 24h"
        triage.escalation_required = severity in ("HIGH", "CRITICAL")

        diagnosis = DiagnosisResult()
        diagnosis.differentials = [
            DiagnosisCandidate(
                icd_code="CA80",
                condition_name="Acute upper respiratory infection",
                probability="high",
                confidence_score=0.8,
                evidence_citations=["NICE CKS URI 2023"],
            )
        ]
        return profile, triage, diagnosis

    def test_disclaimer_in_patient_report(self):
        profile, triage, diagnosis = self._build_inputs("LOW")
        report = self.generator.generate(profile, triage, diagnosis)
        self.assertIn("informational purposes only", report.patient_report)

    def test_no_medication_dosages_in_report(self):
        import re
        profile, triage, diagnosis = self._build_inputs("LOW")
        report = self.generator.generate(profile, triage, diagnosis)
        self.assertIsNone(re.search(r"\d+\s*mg", report.patient_report, re.IGNORECASE))

    def test_safety_gate_blocks_dosage(self):
        from agent.modules.report_generator import MedicalReport
        profile, triage, diagnosis = self._build_inputs("LOW")
        report = MedicalReport(session_id="test", generated_at="2026-01-01", triage_severity="LOW", patient_report="", doctor_summary={})
        blocked = self.generator._apply_safety_gates("Take 400mg ibuprofen", report)
        self.assertNotIn("400mg", blocked)
        self.assertFalse(report.safety_compliant)

    def test_critical_report_starts_with_emergency(self):
        profile, triage, diagnosis = self._build_inputs("CRITICAL")
        triage.escalation_reason = "stroke FAST criteria"
        triage.red_flags = ["stroke FAST criteria"]
        report = self.generator.generate(profile, triage, diagnosis)
        first_line = report.patient_report.split("\n")[0].lower()
        self.assertTrue("emergency" in first_line or "call" in first_line)

    def test_critical_triage_adds_safety_alert(self):
        profile, triage, diagnosis = self._build_inputs("CRITICAL")
        triage.escalation_reason = "test critical"
        report = self.generator.generate(profile, triage, diagnosis)
        self.assertTrue(len(report.safety_alerts) > 0)

    def test_doctor_summary_contains_triage_scores(self):
        profile, triage, diagnosis = self._build_inputs("MEDIUM")
        report = self.generator.generate(profile, triage, diagnosis)
        self.assertIn("news2_score", report.doctor_summary["triage"])
        self.assertIn("qsofa_score", report.doctor_summary["triage"])


# ─── MemoryManager Tests ─────────────────────────────────────────────────────

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        from agent.memory.memory_manager import MemoryManager
        self.memory = MemoryManager(db_path=self.tmp.name)

    def tearDown(self):
        import os
        try:
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_save_and_get_session(self):
        self.memory.save_session(
            session_id="sess-001",
            triage_severity="HIGH",
            news2_score=6,
            qsofa_score=1,
            curb65_score=0,
            gcs_total=None,
            body_system="cardiovascular",
            red_flags=["chest pain"],
            escalation_required=True,
            top_icd_code="BA80.0",
            top_condition_name="STEMI",
            patient_report_length=200,
            safety_compliant=True,
            llm_provider_used="claude",
            session_data={"test": True},
        )
        session = self.memory.get_session("sess-001")
        self.assertIsNotNone(session)
        self.assertEqual(session["triage_severity"], "HIGH")

    def test_recent_sessions_ordering(self):
        for i in range(3):
            self.memory.save_session(
                session_id=f"sess-{i:03d}", triage_severity="LOW", news2_score=1,
                qsofa_score=0, curb65_score=0, gcs_total=None, body_system="other",
                red_flags=[], escalation_required=False, top_icd_code="", top_condition_name="",
                patient_report_length=100, safety_compliant=True,
                llm_provider_used="none", session_data={},
            )
        sessions = self.memory.get_recent_sessions(10)
        self.assertEqual(len(sessions), 3)

    def test_paper_deduplication(self):
        self.memory.mark_paper_known("https://pubmed.ncbi.nlm.nih.gov/12345/", title="Test paper")
        self.assertTrue(self.memory.is_known_paper("https://pubmed.ncbi.nlm.nih.gov/12345/"))
        self.assertFalse(self.memory.is_known_paper("https://pubmed.ncbi.nlm.nih.gov/99999/"))

    def test_cost_logging(self):
        self.memory.log_llm_cost("claude", "claude-sonnet-4-6", 500, 200, 0.005, "diagnosis")
        summary = self.memory.get_cost_summary()
        self.assertIn("claude", summary)

    def test_get_stats(self):
        stats = self.memory.get_stats()
        self.assertIn("total_sessions", stats)
        self.assertIn("escalations", stats)
        self.assertIn("known_papers", stats)


# ─── LLM Client Tests ────────────────────────────────────────────────────────

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        from tools.llm_client import UnifiedLLMClient
        self.client = UnifiedLLMClient()

    def test_privacy_mode_uses_ollama_chain(self):
        import os
        os.environ["PRIVACY_MODE"] = "true"
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        chain = client._build_provider_chain()
        self.assertEqual(chain, ["ollama"])
        del os.environ["PRIVACY_MODE"]

    def test_no_keys_fallback_to_ollama(self):
        import os
        saved_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        saved_openai = os.environ.pop("OPENAI_API_KEY", None)
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        chain = client._build_provider_chain()
        self.assertIn("ollama", chain)
        if saved_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = saved_anthropic
        if saved_openai:
            os.environ["OPENAI_API_KEY"] = saved_openai

    def test_all_providers_fail_returns_fallback(self):
        client = self.client
        client._anthropic_key = ""
        client._openai_key = ""
        client._ollama_url = "http://nonexistent.invalid:11434"
        result = client.complete("test prompt")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ─── HF Model Manager Tests ──────────────────────────────────────────────────

class TestHFModelManager(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        HFModelManager = __import__("tools.hf_model_manager", fromlist=["HFModelManager"]).HFModelManager
        HFModelManager._instance = None
        self.manager = HFModelManager(models_dir=self.tmp_dir)

    def test_heuristic_ner_finds_terms(self):
        entities = self.manager._heuristic_ner("I have chest pain and fever")
        self.assertIn("chest pain", entities)
        self.assertIn("fever", entities)

    def test_heuristic_ner_no_false_positives(self):
        entities = self.manager._heuristic_ner("the weather is nice today")
        self.assertEqual(len(entities), 0)

    def test_tfidf_fallback_encode_is_normalized(self):
        import math
        vec = self.manager._tfidf_fallback_encode("chest pain diagnosis")
        magnitude = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_heuristic_rerank_orders_by_relevance(self):
        docs = ["irrelevant document", "chest pain diagnosis treatment", "weather forecast"]
        result = self.manager._heuristic_rerank("chest pain", docs, top_k=2)
        self.assertEqual(len(result), 2)
        top_idx = result[0][0]
        self.assertEqual(docs[top_idx], "chest pain diagnosis treatment")


# ─── Integration Tests ───────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    def _get_orchestrator(self):
        from agent.orchestrator import MedicalDiagnosisOrchestrator
        import tempfile
        config = {"memory": {"db_path": tempfile.mktemp(suffix=".db")}}
        return MedicalDiagnosisOrchestrator(config)

    def test_low_acuity_full_pipeline(self):
        import asyncio
        orch = self._get_orchestrator()
        result = asyncio.run(orch.diagnose("mild headache and slight fatigue for 1 day"))
        self.assertIn("triage", result)
        self.assertIn("differentials", result)
        self.assertIn("patient_report", result)
        self.assertGreater(len(result["differentials"]), 0)
        self.assertIn("informational purposes only", result["patient_report"])

    def test_critical_escalation_flag(self):
        import asyncio
        orch = self._get_orchestrator()
        result = asyncio.run(orch.diagnose(
            "crushing chest pain with diaphoresis and severe shortness of breath"
        ))
        self.assertEqual(result["triage"]["severity"], "CRITICAL")
        self.assertTrue(result["triage"]["escalation_required"])

    def test_no_medication_dosage_in_any_output(self):
        import asyncio
        import re
        orch = self._get_orchestrator()
        result = asyncio.run(orch.diagnose("fever and cough for 3 days"))
        self.assertIsNone(re.search(r"\d+\s*mg", result["patient_report"], re.IGNORECASE))

    def test_safety_disclaimer_always_present(self):
        import asyncio
        orch = self._get_orchestrator()
        result = asyncio.run(orch.diagnose("sore throat"))
        self.assertIn("informational", result["patient_report"].lower())

    def test_stats_returns_counts(self):
        import asyncio
        orch = self._get_orchestrator()
        asyncio.run(orch.diagnose("test symptoms"))
        stats = orch.get_stats()
        self.assertGreaterEqual(stats["total_sessions"], 1)


# ─── CLI Smoke Tests ─────────────────────────────────────────────────────────

class TestCLISmoke(unittest.TestCase):
    def _run_cli(self, args: list[str]):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        return runner.invoke(cli, args)

    def test_cli_diagnose_json_output(self):
        result = self._run_cli(["diagnose", "mild headache", "--output", "json"])
        self.assertEqual(result.exit_code, 0)
        try:
            data = json.loads(result.output)
            self.assertIn("triage", data)
        except json.JSONDecodeError:
            self.fail("CLI did not return valid JSON")

    def test_cli_stats(self):
        result = self._run_cli(["stats"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("total_sessions", data)

    def test_cli_help(self):
        result = self._run_cli(["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("diagnose", result.output)

    def test_cli_diagnose_empty_fails(self):
        result = self._run_cli(["diagnose", ""])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_diagnose_markdown_output(self):
        result = self._run_cli(["diagnose", "chest pain since 1 hour", "--output", "markdown"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Triage", result.output)


if __name__ == "__main__":
    unittest.main()
