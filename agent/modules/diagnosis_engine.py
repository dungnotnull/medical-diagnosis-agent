"""ICD-11 differential diagnosis engine with evidence retrieval and LLM synthesis."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agent.modules.symptom_analyzer import SymptomProfile
from agent.modules.triage_engine import TriageResult, SeverityLevel

logger = logging.getLogger(__name__)

DIFFERENTIAL_SYNTHESIS_PROMPT = """You are a clinical decision support system. Based on the symptom profile and triage assessment below, generate a ranked differential diagnosis list using ICD-11 codes.

Symptom Profile:
{symptom_json}

Triage Result:
{triage_json}

Relevant Evidence:
{evidence}

Return ONLY a JSON array of differential diagnoses (3-5 entries), ordered by probability:
[
  {{
    "icd_code": "ICD-11 code (e.g., BA80.0)",
    "condition_name": "Full condition name",
    "probability": "high|medium|low",
    "confidence_score": 0.0,
    "supporting_features": ["feature from the case"],
    "evidence_citations": ["citation or guideline"],
    "urgent": true
  }}
]

CRITICAL RULES:
- Do NOT recommend medications or dosages
- Do NOT state a definitive diagnosis — these are differentials only
- All urgent=true entries must have evidence citations
- Confidence scores must be between 0.0 and 1.0"""

ICD11_COMMON_PRESENTATIONS = {
    "cardiovascular": [
        {"icd_code": "BA80.0", "condition_name": "Acute ST-elevation myocardial infarction", "probability": "high", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["ESC Guidelines STEMI 2023"], "urgent": True},
        {"icd_code": "BA80.1", "condition_name": "Acute non-ST-elevation myocardial infarction", "probability": "high", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["ESC Guidelines NSTEMI 2020"], "urgent": True},
        {"icd_code": "BD10", "condition_name": "Pulmonary embolism", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["ESC PE Guidelines 2019"], "urgent": True},
        {"icd_code": "BA91", "condition_name": "Stable angina pectoris", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["ESC Chronic Coronary Syndrome 2019"], "urgent": False},
    ],
    "respiratory": [
        {"icd_code": "CA40.0", "condition_name": "Pneumonia, unspecified", "probability": "high", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["BTS CAP Guidelines 2021"], "urgent": True},
        {"icd_code": "CA22", "condition_name": "Acute exacerbation of COPD", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["GOLD COPD Guidelines 2024"], "urgent": True},
        {"icd_code": "CB23", "condition_name": "Acute asthma attack", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["BTS/SIGN Asthma Guidelines 2023"], "urgent": True},
        {"icd_code": "CA80", "condition_name": "Acute upper respiratory infection", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["NICE CKS URI 2023"], "urgent": False},
    ],
    "neurological": [
        {"icd_code": "8B20", "condition_name": "Ischaemic stroke", "probability": "high", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["ESO Stroke Guidelines 2021"], "urgent": True},
        {"icd_code": "8B21", "condition_name": "Intracerebral haemorrhage", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["AHA/ASA ICH Guidelines 2022"], "urgent": True},
        {"icd_code": "8B00", "condition_name": "Subarachnoid haemorrhage", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["NICE Stroke CG68"], "urgent": True},
        {"icd_code": "8A80", "condition_name": "Migraine with aura", "probability": "low", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["IHS ICHD-3 Classification"], "urgent": False},
        {"icd_code": "1C82", "condition_name": "Bacterial meningitis", "probability": "low", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["NICE CG102 Meningitis"], "urgent": True},
    ],
    "gastrointestinal": [
        {"icd_code": "DA92", "condition_name": "Acute appendicitis", "probability": "high", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["WSES Appendicitis Guidelines 2020"], "urgent": True},
        {"icd_code": "DA94", "condition_name": "Peritonitis", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["WSES Peritonitis Guidelines 2023"], "urgent": True},
        {"icd_code": "DB32", "condition_name": "Peptic ulcer disease", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["NICE CKS Peptic Ulcer 2023"], "urgent": False},
        {"icd_code": "DC30", "condition_name": "Acute gastroenteritis", "probability": "high", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["PHE Gastroenteritis Guidelines"], "urgent": False},
    ],
    "other": [
        {"icd_code": "1C00", "condition_name": "Viral illness, unspecified", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["NICE CKS Viral infection"], "urgent": False},
        {"icd_code": "5A00", "condition_name": "Anxiety disorder", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["NICE CG113 Anxiety"], "urgent": False},
        {"icd_code": "MG20", "condition_name": "Sepsis", "probability": "medium", "confidence_score": 0.0, "supporting_features": [], "evidence_citations": ["Surviving Sepsis Campaign 2021"], "urgent": True},
    ],
}


@dataclass
class DiagnosisCandidate:
    icd_code: str
    condition_name: str
    probability: str
    confidence_score: float
    supporting_features: list[str] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)
    urgent: bool = False


@dataclass
class DiagnosisResult:
    differentials: list[DiagnosisCandidate] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)
    disclaimer: str = (
        "These are differential diagnoses for informational purposes only. "
        "A qualified healthcare provider must make any clinical diagnosis."
    )


class DiagnosisEngine:
    def __init__(self, llm_client=None, hf_manager=None, memory_manager=None):
        self._llm = llm_client
        self._hf = hf_manager
        self._memory = memory_manager
        self._faiss_index = None
        self._knowledge_entries: list[dict] = []

    def diagnose(self, profile: SymptomProfile, triage: TriageResult) -> DiagnosisResult:
        result = DiagnosisResult()

        evidence = self._retrieve_evidence(profile)
        result.evidence_used = [e.get("title", "") for e in evidence[:5]]

        differentials = self._run_llm_diagnosis(profile, triage, evidence)
        if differentials:
            result.differentials = differentials
        else:
            result.differentials = self._heuristic_diagnosis(profile, triage)

        result.differentials = self._validate_differentials(result.differentials)
        return result

    def _retrieve_evidence(self, profile: SymptomProfile) -> list[dict]:
        query = f"{profile.body_system} {' '.join(profile.extracted_entities[:3])} diagnosis"
        evidence = []

        if self._hf is not None:
            try:
                evidence = self._hf.retrieve_from_knowledge_brain(query, top_k=5)
            except Exception as e:
                logger.warning("HF knowledge retrieval failed: %s", e)

        if not evidence:
            evidence = self._keyword_evidence_fallback(query)
        return evidence

    def _keyword_evidence_fallback(self, query: str) -> list[dict]:
        query_lower = query.lower()
        fallback_evidence = {
            "cardiovascular": [
                {"title": "ESC Guidelines STEMI 2023", "abstract": "Immediate PCI is the preferred reperfusion strategy."},
                {"title": "AHA Chest Pain Guidelines 2021", "abstract": "High-sensitivity troponin is recommended for chest pain evaluation."},
            ],
            "respiratory": [
                {"title": "BTS CAP Guidelines 2021", "abstract": "CURB-65 recommended for pneumonia severity assessment."},
                {"title": "GOLD COPD 2024", "abstract": "Exacerbations treated with bronchodilators and systemic corticosteroids."},
            ],
            "neurological": [
                {"title": "ESO Stroke Guidelines 2021", "abstract": "IV alteplase within 4.5h for eligible ischaemic stroke patients."},
                {"title": "NICE Meningitis CG102", "abstract": "Immediate IV antibiotics if bacterial meningitis suspected."},
            ],
        }
        for system, entries in fallback_evidence.items():
            if system in query_lower:
                return entries
        return [{"title": "WHO ICD-11 Classification", "abstract": "International standard for disease classification."}]

    def _run_llm_diagnosis(
        self, profile: SymptomProfile, triage: TriageResult, evidence: list[dict]
    ) -> Optional[list[DiagnosisCandidate]]:
        if self._llm is None:
            return None
        try:
            evidence_text = "\n".join(
                f"- {e.get('title', 'Source')}: {e.get('abstract', '')[:300]}"
                for e in evidence[:5]
            )
            symptom_json = json.dumps({
                "onset": profile.onset,
                "quality": profile.quality,
                "severity": profile.severity,
                "body_system": profile.body_system,
                "associated_symptoms": profile.associated_symptoms,
                "red_flags": profile.red_flag_keywords,
            }, indent=2)
            triage_json = json.dumps({
                "severity": triage.severity.value,
                "news2_score": triage.news2_score,
                "qsofa_score": triage.qsofa_score,
                "curb65_score": triage.curb65_score,
                "gcs_total": triage.gcs_total,
                "red_flags": triage.red_flags,
            }, indent=2)
            prompt = DIFFERENTIAL_SYNTHESIS_PROMPT.format(
                symptom_json=symptom_json,
                triage_json=triage_json,
                evidence=evidence_text,
            )
            response = self._llm.complete(prompt=prompt, max_tokens=1200, temperature=0.1)
            return self._parse_differentials(response)
        except Exception as e:
            logger.warning("LLM diagnosis failed: %s — using heuristic", e)
            return None

    def _parse_differentials(self, text: str) -> Optional[list[DiagnosisCandidate]]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
                candidates = []
                for item in items:
                    if isinstance(item, dict):
                        candidates.append(DiagnosisCandidate(
                            icd_code=item.get("icd_code", "XX"),
                            condition_name=item.get("condition_name", "Unknown"),
                            probability=item.get("probability", "low"),
                            confidence_score=float(item.get("confidence_score", 0.3)),
                            supporting_features=item.get("supporting_features", []),
                            evidence_citations=item.get("evidence_citations", []),
                            urgent=bool(item.get("urgent", False)),
                        ))
                return candidates if candidates else None
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _heuristic_diagnosis(self, profile: SymptomProfile, triage: TriageResult) -> list[DiagnosisCandidate]:
        system = profile.body_system if profile.body_system in ICD11_COMMON_PRESENTATIONS else "other"
        candidates_data = ICD11_COMMON_PRESENTATIONS.get(system, ICD11_COMMON_PRESENTATIONS["other"])

        candidates = []
        for idx, cd in enumerate(candidates_data[:5]):
            cd_copy = dict(cd)
            cd_copy["supporting_features"] = profile.extracted_entities[:2]
            if triage.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
                cd_copy["confidence_score"] = max(0.6, 0.8 - idx * 0.1)
            else:
                cd_copy["confidence_score"] = max(0.3, 0.6 - idx * 0.1)
            candidates.append(DiagnosisCandidate(**cd_copy))

        return candidates[:4]

    def _validate_differentials(self, candidates: list[DiagnosisCandidate]) -> list[DiagnosisCandidate]:
        validated = []
        for c in candidates:
            if not c.evidence_citations:
                c.evidence_citations = ["WHO ICD-11 Classification 2022"]
            c.confidence_score = max(0.0, min(1.0, c.confidence_score))
            validated.append(c)
        return validated[:5]
