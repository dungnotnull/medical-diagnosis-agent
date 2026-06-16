"""Clinical triage engine: NEWS2, qSOFA, CURB-65, GCS scoring and red-flag detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agent.modules.symptom_analyzer import SymptomProfile

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class VitalSigns:
    respiration_rate: Optional[int] = None   # breaths/min
    spo2: Optional[float] = None             # %
    supplemental_o2: bool = False
    systolic_bp: Optional[int] = None        # mmHg
    pulse: Optional[int] = None              # bpm
    consciousness: Optional[str] = None      # A/C/V/P/U (ACVPU)
    temperature: Optional[float] = None      # Celsius
    gcs_eye: Optional[int] = None            # 1-4
    gcs_verbal: Optional[int] = None         # 1-5
    gcs_motor: Optional[int] = None          # 1-6
    urea: Optional[float] = None             # mmol/L
    age: Optional[int] = None


@dataclass
class TriageResult:
    news2_score: int = 0
    news2_partial: bool = True
    qsofa_score: int = 0
    curb65_score: int = 0
    gcs_total: Optional[int] = None
    severity: SeverityLevel = SeverityLevel.LOW
    red_flags: list[str] = field(default_factory=list)
    escalation_required: bool = False
    escalation_reason: str = ""
    scoring_notes: list[str] = field(default_factory=list)
    recommended_action: str = ""


# Red-flag pattern definitions
RED_FLAG_PATTERNS = {
    "chest_pain_diaphoresis": {
        "keywords": [["chest pain", "chest tightness"], ["sweating", "diaphoresis", "cold sweat"]],
        "reason": "Chest pain with diaphoresis — possible acute coronary syndrome (STEMI/NSTEMI)",
        "severity": SeverityLevel.CRITICAL,
    },
    "stroke_fast": {
        "keywords": [["facial droop", "face drooping", "face drop", "slurred speech",
                       "arm weakness", "arm numb", "sudden speech", "sudden weakness",
                       "sudden vision loss", "thunderclap headache", "worst headache of my life"]],
        "reason": "FAST stroke criteria positive — possible stroke/TIA",
        "severity": SeverityLevel.CRITICAL,
    },
    "respiratory_distress": {
        "keywords": [["can't breathe", "cannot breathe", "not breathing", "gasping",
                       "respiratory distress", "severe shortness of breath",
                       "unable to speak in full sentences", "accessory muscle", "cyanosis"]],
        "reason": "Severe respiratory distress — immediate airway support needed",
        "severity": SeverityLevel.CRITICAL,
    },
    "altered_consciousness": {
        "keywords": [["unconscious", "unresponsive", "confusion", "agitation",
                       "drowsy", "GCS"]],
        "reason": "Altered level of consciousness — possible serious neurological event",
        "severity": SeverityLevel.CRITICAL,
    },
    "meningism": {
        "keywords": [["neck stiffness", "photophobia", "petechial rash",
                       "purpuric rash", "meningism"]],
        "reason": "Meningism signs — possible meningococcal meningitis (life-threatening)",
        "severity": SeverityLevel.CRITICAL,
    },
    "severe_abdominal": {
        "keywords": [["rigid abdomen", "board-like abdomen", "sudden severe abdominal pain",
                       "tearing abdominal pain", "ripping pain"]],
        "reason": "Possible peritonitis or aortic aneurysm — surgical emergency",
        "severity": SeverityLevel.HIGH,
    },
    "haemorrhage": {
        "keywords": [["coughing blood", "haemoptysis", "vomiting blood", "haematemesis",
                       "black tarry stool", "melaena", "severe rectal bleeding"]],
        "reason": "Active haemorrhage — requires urgent investigation",
        "severity": SeverityLevel.HIGH,
    },
    "sepsis_signs": {
        "keywords": [["high fever", "confusion", "rapid breathing", "suspected infection"]],
        "reason": "Signs consistent with sepsis — time-critical intervention",
        "severity": SeverityLevel.HIGH,
    },
}


class TriageEngine:
    def score(self, profile: SymptomProfile, vitals: Optional[VitalSigns] = None) -> TriageResult:
        result = TriageResult()
        result.red_flags = list(profile.red_flag_keywords)
        self._detect_red_flags(profile.raw_text, result)

        if vitals:
            result.news2_score = self._compute_news2(vitals, result)
            result.news2_partial = False
            result.qsofa_score = self._compute_qsofa(vitals, profile, result)
            result.curb65_score = self._compute_curb65(vitals, profile, result)
            result.gcs_total = self._compute_gcs(vitals, result)
        else:
            result.scoring_notes.append("Vitals not provided — NEWS2 estimated from symptom text only")
            self._estimate_severity_from_text(profile, result)

        result.severity = self._determine_overall_severity(result)
        result.escalation_required = result.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
        result.recommended_action = self._build_recommended_action(result)
        return result

    def _detect_red_flags(self, text: str, result: TriageResult) -> None:
        text_lower = text.lower()
        for pattern_name, pattern_def in RED_FLAG_PATTERNS.items():
            triggered = False
            for keyword_group in pattern_def["keywords"]:
                if any(kw in text_lower for kw in keyword_group):
                    triggered = True
                    break
            if triggered:
                flag_text = pattern_def["reason"]
                if flag_text not in result.red_flags:
                    result.red_flags.append(flag_text)
                if pattern_def["severity"] == SeverityLevel.CRITICAL:
                    result.escalation_required = True
                    result.escalation_reason = pattern_def["reason"]

    def _compute_news2(self, v: VitalSigns, result: TriageResult) -> int:
        score = 0

        if v.respiration_rate is not None:
            rr = v.respiration_rate
            if rr <= 8:
                score += 3
            elif rr <= 11:
                score += 1
            elif rr <= 20:
                score += 0
            elif rr <= 24:
                score += 2
            else:
                score += 3
            result.scoring_notes.append(f"RR={rr} → +{score} NEWS2 pts")

        if v.spo2 is not None:
            if v.supplemental_o2:
                if v.spo2 >= 97:
                    score += 3
                elif v.spo2 >= 95:
                    score += 2
                elif v.spo2 >= 93:
                    score += 1
            else:
                if v.spo2 >= 96:
                    score += 0
                elif v.spo2 >= 94:
                    score += 1
                elif v.spo2 >= 92:
                    score += 2
                else:
                    score += 3

        if v.systolic_bp is not None:
            sbp = v.systolic_bp
            if sbp <= 90:
                score += 3
            elif sbp <= 100:
                score += 2
            elif sbp <= 110:
                score += 1
            elif sbp <= 219:
                score += 0
            else:
                score += 3

        if v.pulse is not None:
            hr = v.pulse
            if hr <= 40:
                score += 3
            elif hr <= 50:
                score += 1
            elif hr <= 90:
                score += 0
            elif hr <= 110:
                score += 1
            elif hr <= 130:
                score += 2
            else:
                score += 3

        if v.consciousness is not None:
            acvpu = v.consciousness.upper()
            if acvpu == "A":
                score += 0
            elif acvpu in ("C", "V", "P", "U"):
                score += 3

        if v.temperature is not None:
            temp = v.temperature
            if temp <= 35.0:
                score += 3
            elif temp <= 36.0:
                score += 1
            elif temp <= 38.0:
                score += 0
            elif temp <= 39.0:
                score += 1
            else:
                score += 2

        return score

    def _compute_qsofa(self, v: VitalSigns, profile: SymptomProfile, result: TriageResult) -> int:
        score = 0
        text_lower = profile.raw_text.lower()

        if v.consciousness and v.consciousness.upper() != "A":
            score += 1
        elif any(kw in text_lower for kw in ["confusion", "altered", "disoriented"]):
            score += 1

        if v.respiration_rate and v.respiration_rate >= 22:
            score += 1

        if v.systolic_bp and v.systolic_bp <= 100:
            score += 1

        if score >= 2:
            result.red_flags.append(f"qSOFA ≥2 ({score}/3) — possible sepsis; urgent review required")
            result.scoring_notes.append(f"qSOFA={score}: sepsis risk elevated")
        return score

    def _compute_curb65(self, v: VitalSigns, profile: SymptomProfile, result: TriageResult) -> int:
        score = 0
        text_lower = profile.raw_text.lower()

        if any(kw in text_lower for kw in ["confusion", "confused", "disoriented"]):
            score += 1

        if v.urea is not None and v.urea > 7.0:
            score += 1

        if v.respiration_rate is not None and v.respiration_rate >= 30:
            score += 1

        if v.systolic_bp is not None and (v.systolic_bp < 90 or (
                v.systolic_bp <= 60 if hasattr(v, 'diastolic_bp') else False)):
            score += 1

        if v.age is not None and v.age >= 65:
            score += 1

        if score >= 3:
            result.scoring_notes.append(f"CURB-65={score}: severe pneumonia risk — consider hospital admission")
        return score

    def _compute_gcs(self, v: VitalSigns, result: TriageResult) -> Optional[int]:
        if v.gcs_eye is None and v.gcs_verbal is None and v.gcs_motor is None:
            return None
        eye = v.gcs_eye or 4
        verbal = v.gcs_verbal or 5
        motor = v.gcs_motor or 6
        total = eye + verbal + motor
        if total <= 8:
            result.red_flags.append(f"GCS={total} (≤8): severe impairment — emergency airway management needed")
            result.escalation_required = True
        elif total <= 12:
            result.scoring_notes.append(f"GCS={total}: moderate impairment — close monitoring required")
        return total

    def _estimate_severity_from_text(self, profile: SymptomProfile, result: TriageResult) -> None:
        if profile.is_immediately_critical() or result.escalation_required:
            result.news2_score = 7
            result.scoring_notes.append("Critical keywords detected — NEWS2 estimated HIGH-RISK (≥7)")
        elif profile.has_red_flags():
            result.news2_score = 5
            result.scoring_notes.append("Red flag keywords present — NEWS2 estimated MEDIUM-HIGH (5)")
        else:
            text_lower = profile.raw_text.lower()
            high_risk = ["severe", "worst ever", "sudden onset", "can't function"]
            medium_risk = ["moderate", "quite bad", "getting worse"]
            if any(kw in text_lower for kw in high_risk):
                result.news2_score = 4
            elif any(kw in text_lower for kw in medium_risk):
                result.news2_score = 2
            else:
                result.news2_score = 1

    def _determine_overall_severity(self, result: TriageResult) -> SeverityLevel:
        if result.escalation_required:
            return SeverityLevel.CRITICAL

        if result.news2_score >= 7 or result.qsofa_score >= 2 or result.curb65_score >= 3:
            return SeverityLevel.CRITICAL
        if result.news2_score >= 5:
            return SeverityLevel.HIGH
        if result.news2_score >= 3 or result.curb65_score == 2:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def _build_recommended_action(self, result: TriageResult) -> str:
        if result.severity == SeverityLevel.CRITICAL:
            return (
                "CALL EMERGENCY SERVICES (911/999) IMMEDIATELY. "
                "Do not wait. This presentation requires immediate emergency care."
            )
        if result.severity == SeverityLevel.HIGH:
            return (
                "Seek urgent medical attention NOW — go to the nearest emergency department "
                "or call your doctor for immediate assessment. Do not drive yourself."
            )
        if result.severity == SeverityLevel.MEDIUM:
            return (
                "Contact your GP or urgent care centre within the next few hours. "
                "If symptoms worsen significantly, go to A&E."
            )
        return (
            "Monitor symptoms at home. Schedule a GP appointment within 24–48 hours if symptoms persist. "
            "Seek emergency care if symptoms suddenly worsen."
        )
