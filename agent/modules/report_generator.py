"""Report generator with safety gates: patient report and doctor summary."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agent.modules.symptom_analyzer import SymptomProfile
from agent.modules.triage_engine import TriageResult, SeverityLevel
from agent.modules.diagnosis_engine import DiagnosisResult

logger = logging.getLogger(__name__)

SAFETY_DISCLAIMER = (
    "\n\n---\n"
    "**IMPORTANT DISCLAIMER**: This assessment is for informational purposes only and does "
    "NOT replace medical advice, diagnosis, or treatment from a qualified healthcare provider. "
    "If you are experiencing a medical emergency, call emergency services (911/999) immediately."
)

PATIENT_GUIDANCE_PROMPT = """You are a compassionate medical information assistant.
Provide clear, plain-language guidance for a patient based on their assessment below.

Triage Level: {triage_level}
Recommended Action: {recommended_action}
Top Differential Conditions (informational only): {top_conditions}
Red Flags Present: {red_flags}

Write a patient-friendly summary (150-200 words) that:
1. States clearly what to do RIGHT NOW (call 911, go to ED, see GP, monitor at home)
2. Explains in plain language what the symptoms might suggest (general terms, no definitive diagnoses)
3. Lists 3-4 safe self-monitoring steps (e.g., rest, note any changes, keep record)
4. States when to escalate to emergency care

CRITICAL RULES:
- NEVER mention specific medication names or dosages
- NEVER state a definitive diagnosis
- Use simple language (reading level 8th grade)
- For CRITICAL triage: first sentence must be 'Call emergency services (911/999) immediately.'
"""

DOSAGE_PATTERNS = [
    r"\b\d+\s*mg\b",
    r"\b\d+\s*mcg\b",
    r"\b\d+\s*ml\b",
    r"\btake\s+\d+",
    r"\b\d+\s*tablet",
    r"\b\d+\s*dose",
    r"\bprescribe\b",
    r"\bprescription\b",
    r"\badminister\s+\d+",
]

MEDICATION_NAME_PATTERNS = [
    r"\b(aspirin|ibuprofen|paracetamol|acetaminophen|morphine|codeine|amoxicillin|"
    r"penicillin|metformin|atorvastatin|lisinopril|metoprolol|warfarin|heparin)\b",
]


@dataclass
class MedicalReport:
    session_id: str
    generated_at: str
    triage_severity: str
    patient_report: str
    doctor_summary: dict
    safety_alerts: list[str] = field(default_factory=list)
    safety_compliant: bool = True
    escalation_required: bool = False
    word_count: int = 0


class ReportGenerator:
    def __init__(self, llm_client=None, hf_manager=None):
        self._llm = llm_client
        self._hf = hf_manager

    def generate(
        self,
        profile: SymptomProfile,
        triage: TriageResult,
        diagnosis: DiagnosisResult,
        session_id: str = "unknown",
    ) -> MedicalReport:
        report = MedicalReport(
            session_id=session_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            triage_severity=triage.severity.value,
            patient_report="",
            doctor_summary={},
        )

        report.escalation_required = triage.escalation_required
        if triage.severity == SeverityLevel.CRITICAL:
            report.safety_alerts.append(
                "CRITICAL: Patient requires immediate emergency services. "
                "Call 911/999 now. Do not delay."
            )

        patient_text = self._generate_patient_guidance(profile, triage, diagnosis)
        patient_text = self._apply_safety_gates(patient_text, report)
        report.patient_report = patient_text + SAFETY_DISCLAIMER
        report.word_count = len(report.patient_report.split())

        report.doctor_summary = self._build_doctor_summary(profile, triage, diagnosis)
        return report

    def _generate_patient_guidance(
        self, profile: SymptomProfile, triage: TriageResult, diagnosis: DiagnosisResult
    ) -> str:
        if triage.severity == SeverityLevel.CRITICAL:
            return self._critical_guidance(triage)

        top_conditions = [d.condition_name for d in diagnosis.differentials[:3]]

        if self._llm is not None:
            try:
                prompt = PATIENT_GUIDANCE_PROMPT.format(
                    triage_level=triage.severity.value,
                    recommended_action=triage.recommended_action,
                    top_conditions=", ".join(top_conditions) if top_conditions else "under evaluation",
                    red_flags=", ".join(triage.red_flags[:3]) if triage.red_flags else "none identified",
                )
                response = self._llm.complete(prompt=prompt, max_tokens=400, temperature=0.2)
                if response and len(response.strip()) > 50:
                    return response.strip()
            except Exception as e:
                logger.warning("LLM patient guidance failed: %s", e)

        return self._template_guidance(triage, top_conditions)

    def _critical_guidance(self, triage: TriageResult) -> str:
        reason = triage.escalation_reason or "life-threatening symptoms detected"
        return (
            f"**Call emergency services (911/999) immediately.**\n\n"
            f"Based on your reported symptoms, this situation requires urgent emergency care. "
            f"Reason: {reason}\n\n"
            f"While waiting for emergency services:\n"
            f"- Stay as calm and still as possible\n"
            f"- Do not eat or drink anything\n"
            f"- Have someone stay with you\n"
            f"- Unlock your door so paramedics can enter\n\n"
            f"Do NOT drive yourself to the hospital."
        )

    def _template_guidance(self, triage: TriageResult, top_conditions: list[str]) -> str:
        action_map = {
            SeverityLevel.HIGH: (
                "**Please go to your nearest Emergency Department or Urgent Care centre now**, "
                "or call your doctor immediately for same-day assessment."
            ),
            SeverityLevel.MEDIUM: (
                "**Contact your GP or an urgent care service within the next few hours.** "
                "If symptoms worsen significantly before then, go to A&E."
            ),
            SeverityLevel.LOW: (
                "**Monitor your symptoms at home** and schedule a GP appointment within 24-48 hours "
                "if symptoms persist or worsen."
            ),
        }
        action = action_map.get(triage.severity, triage.recommended_action)
        conditions_text = ""
        if top_conditions:
            conditions_text = (
                f"\n\nYour symptoms may be consistent with conditions such as: "
                f"{', '.join(top_conditions[:2])}. "
                f"These are possibilities only — a doctor must make any formal diagnosis."
            )

        return (
            f"{action}{conditions_text}\n\n"
            f"**Self-monitoring steps:**\n"
            f"- Note any changes in your symptoms (better, worse, or new symptoms)\n"
            f"- Record your temperature if possible\n"
            f"- Rest and stay well-hydrated\n"
            f"- Call emergency services if symptoms suddenly become much worse"
        )

    def _apply_safety_gates(self, text: str, report: MedicalReport) -> str:
        for pattern in DOSAGE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Safety gate blocked dosage information in report")
                report.safety_alerts.append("Safety gate: medication dosage information was removed")
                report.safety_compliant = False
                text = re.sub(pattern, "[dosage information removed]", text, flags=re.IGNORECASE)

        for pattern in MEDICATION_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Safety gate blocked specific medication name in report")
                report.safety_alerts.append("Safety gate: specific medication name was removed")
                report.safety_compliant = False
                text = re.sub(pattern, "[medication name removed]", text, flags=re.IGNORECASE)

        if "definitive diagnosis" not in text.lower() and re.search(r"\bdiagnosis(?:ed)?\s+with\b", text, re.IGNORECASE):
            logger.warning("Safety gate blocked definitive diagnosis statement")
            report.safety_compliant = False

        return text

    def _build_doctor_summary(
        self, profile: SymptomProfile, triage: TriageResult, diagnosis: DiagnosisResult
    ) -> dict:
        return {
            "session_summary": {
                "presentation": profile.raw_text[:500],
                "opqrst": {
                    "onset": profile.onset,
                    "provocation": profile.provocation,
                    "quality": profile.quality,
                    "radiation": profile.radiation,
                    "severity": profile.severity,
                    "time": profile.time,
                },
                "associated_symptoms": profile.associated_symptoms,
                "extracted_entities": profile.extracted_entities,
                "body_system": profile.body_system,
            },
            "triage": {
                "severity": triage.severity.value,
                "news2_score": triage.news2_score,
                "qsofa_score": triage.qsofa_score,
                "curb65_score": triage.curb65_score,
                "gcs_total": triage.gcs_total,
                "red_flags": triage.red_flags,
                "escalation_required": triage.escalation_required,
                "escalation_reason": triage.escalation_reason,
                "scoring_notes": triage.scoring_notes,
            },
            "differentials": [
                {
                    "icd_code": d.icd_code,
                    "condition_name": d.condition_name,
                    "probability": d.probability,
                    "confidence_score": d.confidence_score,
                    "urgent": d.urgent,
                    "evidence_citations": d.evidence_citations,
                }
                for d in diagnosis.differentials
            ],
            "evidence_used": diagnosis.evidence_used,
            "disclaimer": diagnosis.disclaimer,
        }

    def format_patient_markdown(self, report: MedicalReport) -> str:
        lines = [
            f"# Medical Assessment Report",
            f"**Generated:** {report.generated_at}",
            f"**Severity Level:** {report.triage_severity}",
            "",
        ]
        if report.safety_alerts:
            lines.extend(["## ⚠️ ALERTS", ""])
            for alert in report.safety_alerts:
                lines.append(f"> **{alert}**")
            lines.append("")

        lines.extend(["## Assessment", "", report.patient_report])
        return "\n".join(lines)
