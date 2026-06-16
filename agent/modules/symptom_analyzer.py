"""Symptom extraction and OPQRST structured interview module."""
from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

BODY_SYSTEMS = [
    "cardiovascular", "respiratory", "neurological", "gastrointestinal",
    "musculoskeletal", "genitourinary", "dermatological", "psychiatric",
    "endocrine", "haematological", "other",
]

RED_FLAG_KEYWORDS = [
    "chest pain", "crushing", "can't breathe", "cannot breathe", "shortness of breath",
    "difficulty breathing", "unconscious", "unresponsive", "seizure", "stroke",
    "slurred speech", "facial droop", "arm weakness", "sudden severe headache",
    "worst headache", "thunderclap", "coughing blood", "vomiting blood",
    "black tarry stool", "sudden vision loss", "severe abdominal pain",
    "rigidity", "neck stiffness", "high fever", "confusion", "altered consciousness",
    "not breathing", "no pulse", "severe bleeding", "suicidal",
]

SEVERITY_KEYWORDS_CRITICAL = [
    "crushing", "can't breathe", "not breathing", "unconscious", "unresponsive",
    "no pulse", "severe bleeding", "suicidal",
]

OPQRST_SYSTEM_PROMPT = """You are a clinical triage assistant conducting a structured symptom assessment.
Extract an OPQRST symptom profile from the patient text.

Return ONLY valid JSON with these exact fields:
{
  "onset": "when it started (e.g., '2 hours ago', 'this morning', 'gradual over 3 days')",
  "provocation": "what makes it worse or better (or 'unknown' if not mentioned)",
  "quality": "character of symptom (e.g., 'sharp', 'dull', 'burning', 'pressure', 'throbbing')",
  "radiation": "does it spread anywhere (or 'none' if not mentioned)",
  "severity": "pain scale 0-10 or descriptive severity if not pain",
  "time": "temporal pattern (e.g., 'continuous', 'intermittent every 5 min', 'worsening')",
  "associated_symptoms": ["comma", "separated", "list"],
  "body_system": "one of: cardiovascular|respiratory|neurological|gastrointestinal|musculoskeletal|genitourinary|dermatological|psychiatric|endocrine|haematological|other",
  "red_flag_keywords": ["any", "concerning", "terms", "found"],
  "patient_age_approx": "approximate age if mentioned, else null",
  "patient_sex": "male|female|unknown"
}

IMPORTANT: Do not suggest diagnoses or medications. Extract symptoms only."""


@dataclass
class SymptomProfile:
    raw_text: str
    onset: str = "unknown"
    provocation: str = "unknown"
    quality: str = "unknown"
    radiation: str = "none"
    severity: str = "unknown"
    time: str = "unknown"
    associated_symptoms: list[str] = field(default_factory=list)
    body_system: str = "other"
    red_flag_keywords: list[str] = field(default_factory=list)
    patient_age_approx: Optional[str] = None
    patient_sex: str = "unknown"
    extracted_entities: list[str] = field(default_factory=list)
    opqrst_completeness: float = 0.0

    def is_valid(self) -> bool:
        core_fields = [self.onset, self.quality, self.severity, self.time]
        populated = sum(1 for f in core_fields if f not in ("unknown", "none", ""))
        return populated >= 2 and len(self.extracted_entities) >= 1

    def has_red_flags(self) -> bool:
        return len(self.red_flag_keywords) > 0

    def is_immediately_critical(self) -> bool:
        combined = self.raw_text.lower()
        return any(kw in combined for kw in SEVERITY_KEYWORDS_CRITICAL)


class SymptomAnalyzer:
    def __init__(self, llm_client=None, hf_manager=None):
        self._llm = llm_client
        self._hf = hf_manager

    def analyze(self, patient_text: str) -> SymptomProfile:
        profile = SymptomProfile(raw_text=patient_text)
        profile.red_flag_keywords = self._detect_red_flags(patient_text)
        profile.extracted_entities = self._extract_entities_heuristic(patient_text)

        if self._hf is not None:
            try:
                entities = self._hf.extract_medical_entities(patient_text)
                if entities:
                    profile.extracted_entities = entities
            except Exception as e:
                logger.warning("HF NER failed: %s", e)

        opqrst = self._run_opqrst_interview(patient_text)
        if opqrst:
            profile.onset = opqrst.get("onset", "unknown")
            profile.provocation = opqrst.get("provocation", "unknown")
            profile.quality = opqrst.get("quality", "unknown")
            profile.radiation = opqrst.get("radiation", "none")
            profile.severity = str(opqrst.get("severity", "unknown"))
            profile.time = opqrst.get("time", "unknown")
            profile.associated_symptoms = opqrst.get("associated_symptoms", [])
            profile.body_system = opqrst.get("body_system", "other")
            if opqrst.get("red_flag_keywords"):
                profile.red_flag_keywords = list(
                    set(profile.red_flag_keywords + opqrst["red_flag_keywords"])
                )
            profile.patient_age_approx = opqrst.get("patient_age_approx")
            profile.patient_sex = opqrst.get("patient_sex", "unknown")

        profile.opqrst_completeness = self._compute_completeness(profile)
        return profile

    def _run_opqrst_interview(self, patient_text: str) -> Optional[dict]:
        if self._llm is None:
            return self._heuristic_opqrst(patient_text)
        try:
            prompt = OPQRST_SYSTEM_PROMPT + f"\n\nPatient text: {patient_text}"
            response = self._llm.complete(
                prompt=prompt,
                max_tokens=600,
                temperature=0.1,
            )
            return self._parse_json_response(response)
        except Exception as e:
            logger.warning("LLM OPQRST failed: %s — using heuristic fallback", e)
            return self._heuristic_opqrst(patient_text)

    def _parse_json_response(self, text: str) -> Optional[dict]:
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _heuristic_opqrst(self, text: str) -> dict:
        text_lower = text.lower()
        onset = "unknown"
        for pattern in [r"(\d+\s*(?:hour|day|week|minute)s?\s*ago)", r"(since\s+\w+)", r"(started\s+\w+)"]:
            m = re.search(pattern, text_lower)
            if m:
                onset = m.group(1)
                break

        severity = "unknown"
        m = re.search(r"(\d+)\s*/?\s*10", text_lower)
        if m:
            severity = f"{m.group(1)}/10"
        elif any(w in text_lower for w in ["severe", "very bad", "worst"]):
            severity = "severe (8-10/10)"
        elif any(w in text_lower for w in ["moderate", "quite bad"]):
            severity = "moderate (4-7/10)"
        elif any(w in text_lower for w in ["mild", "slight", "little"]):
            severity = "mild (1-3/10)"

        quality = "unknown"
        for q in ["sharp", "dull", "burning", "pressure", "crushing", "stabbing", "aching", "throbbing"]:
            if q in text_lower:
                quality = q
                break

        body_system = self._classify_body_system(text_lower)

        return {
            "onset": onset,
            "provocation": "unknown",
            "quality": quality,
            "radiation": "none",
            "severity": severity,
            "time": "unknown",
            "associated_symptoms": [],
            "body_system": body_system,
            "red_flag_keywords": [],
            "patient_age_approx": None,
            "patient_sex": "unknown",
        }

    def _classify_body_system(self, text: str) -> str:
        system_keywords = {
            "cardiovascular": ["chest", "heart", "palpitation", "pounding", "racing", "angina"],
            "respiratory": ["breath", "cough", "wheeze", "lung", "inhale", "exhale", "phlegm"],
            "neurological": ["headache", "dizzy", "vision", "speech", "memory", "seizure", "weakness"],
            "gastrointestinal": ["stomach", "nausea", "vomit", "diarrhea", "constipation", "abdomen", "bowel"],
            "musculoskeletal": ["joint", "muscle", "back", "knee", "shoulder", "fracture", "sprain"],
            "genitourinary": ["urinate", "urine", "kidney", "bladder", "groin"],
            "psychiatric": ["anxious", "depressed", "panic", "suicidal", "hallucin"],
        }
        for system, keywords in system_keywords.items():
            if any(kw in text for kw in keywords):
                return system
        return "other"

    def _detect_red_flags(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = []
        for kw in RED_FLAG_KEYWORDS:
            if kw in text_lower:
                found.append(kw)
        return found

    def _extract_entities_heuristic(self, text: str) -> list[str]:
        symptom_terms = [
            "pain", "ache", "fever", "cough", "nausea", "vomiting", "diarrhea",
            "fatigue", "weakness", "dizziness", "headache", "rash", "swelling",
            "bleeding", "shortness of breath", "chest pain", "palpitations",
            "confusion", "numbness", "tingling", "vision", "hearing",
        ]
        text_lower = text.lower()
        return [term for term in symptom_terms if term in text_lower]

    def _compute_completeness(self, profile: SymptomProfile) -> float:
        fields = [
            profile.onset != "unknown",
            profile.provocation != "unknown",
            profile.quality != "unknown",
            profile.radiation != "none",
            profile.severity != "unknown",
            profile.time != "unknown",
        ]
        return sum(fields) / len(fields)
