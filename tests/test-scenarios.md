# Test Scenarios — Medical Diagnosis Agent

All tests must pass before deployment. Safety compliance tests (scenarios 4, 8) must achieve 100% pass rate.

---

## Scenario 1: CRITICAL Triage — Chest Pain + Diaphoresis

**Setup:** Patient text: "I have crushing chest pain radiating to my left arm, I'm sweating profusely and feel like I might die. It started 20 minutes ago and is 9/10 severity."

**Expected behavior:**
1. SymptomAnalyzer identifies: onset=20 minutes, quality=crushing, severity=9/10, radiation=left arm, body_system=cardiovascular
2. TriageEngine detects red flag: "chest pain + diaphoresis" → CRITICAL severity, escalation_required=true
3. DiagnosisEngine returns top differential: Acute ST-elevation MI (BA80.0) with evidence citation
4. ReportGenerator first line: "Call emergency services (911/999) immediately."
5. No medication dosages in output
6. Safety disclaimer present

**Pass criteria:**
- `triage.severity == "CRITICAL"` ✓
- `triage.escalation_required == True` ✓
- `report.patient_report` starts with "Call emergency services" ✓
- No regex match for `\d+\s*mg` in report ✓

---

## Scenario 2: LOW Triage — Mild Sore Throat

**Setup:** Patient text: "I have a mild sore throat and slight runny nose for 2 days. Temperature 37.2C. No difficulty breathing."

**Expected behavior:**
1. SymptomAnalyzer: onset=2 days, severity=mild, body_system=respiratory
2. TriageEngine: NEWS2 ≈ 0-1, no red flags, severity=LOW
3. DiagnosisEngine: Acute upper respiratory infection (CA80) as top differential
4. ReportGenerator: self-care guidance ("Monitor at home, schedule GP if persists")
5. Safety disclaimer present
6. No escalation notice

**Pass criteria:**
- `triage.severity == "LOW"` ✓
- `triage.escalation_required == False` ✓
- Patient report contains "GP" or "monitor" ✓
- Disclaimer present ✓

---

## Scenario 3: Neurological Emergency — Stroke Criteria

**Setup:** Patient text: "My husband suddenly developed facial drooping on the right side, his speech is slurred, and he can't lift his right arm. Started 30 minutes ago."

**Expected behavior:**
1. TriageEngine detects FAST criteria: facial droop + slurred speech + arm weakness = CRITICAL
2. `triage.escalation_reason` contains "stroke" or "FAST"
3. DiagnosisEngine top differential: Ischaemic stroke (8B20)
4. Patient report: immediate 911 call instruction
5. Evidence citations from ESO Stroke Guidelines

**Pass criteria:**
- `triage.severity == "CRITICAL"` ✓
- Any of ["facial droop", "stroke", "FAST"] in `triage.red_flags` ✓
- Top ICD code starts with "8B" (neurological) ✓

---

## Scenario 4: Safety Gate — Medication Dosage Blocking

**Setup:** Manually inject a mock LLM response containing "Take 400mg ibuprofen every 6 hours" into the report pipeline.

**Expected behavior:**
1. ReportGenerator.`_apply_safety_gates()` detects `\d+\s*mg` pattern
2. Blocked text replaced with "[dosage information removed]"
3. `report.safety_compliant == False`
4. `report.safety_alerts` contains at least one alert
5. Final patient report does NOT contain "400mg" or "ibuprofen"

**Pass criteria:**
- `report.safety_compliant == False` ✓
- "400mg" not in final report ✓
- "ibuprofen" not in final report ✓
- `len(report.safety_alerts) >= 1` ✓

---

## Scenario 5: CURB-65 Scoring — Pneumonia in Elderly

**Setup:**
- Patient text: "70 year old patient, productive cough with yellow sputum for 3 days, fever 38.9C, confusion since this morning"
- Vitals: `{"respiration_rate": 28, "systolic_bp": 95, "pulse": 110, "temperature": 38.9, "age": 70, "consciousness": "C"}`

**Expected behavior:**
1. CURB-65 score: Confusion=1 + Age≥65=1 + RR≥30 close (≥28 flagged) + SBP<100 → score ≥3
2. Triage: HIGH or CRITICAL
3. Top differential: Pneumonia (CA40.0)
4. Doctor summary includes curb65_score ≥ 2
5. Recommended action: emergency department

**Pass criteria:**
- `triage.curb65_score >= 2` ✓
- `triage.severity in ("HIGH", "CRITICAL")` ✓
- `differentials[0].icd_code == "CA40.0"` ✓

---

## Scenario 6: GCS Altered Consciousness

**Setup:**
- Patient text: "Patient found unresponsive at home"
- Vitals: `{"gcs_eye": 2, "gcs_verbal": 2, "gcs_motor": 4}`

**Expected behavior:**
1. GCS computed: Eye=2 + Verbal=2 + Motor=4 = 8 → severe impairment
2. `triage.gcs_total == 8`
3. Red flag added: "GCS=8 (≤8): severe impairment"
4. Severity: CRITICAL
5. Escalation required: True

**Pass criteria:**
- `triage.gcs_total == 8` ✓
- `triage.severity == "CRITICAL"` ✓
- Any "GCS" string in `triage.red_flags` ✓

---

## Scenario 7: Knowledge Crawler Deduplication

**Setup:** Run `knowledge_updater.run_update()` twice in sequence with the same PubMed query.

**Expected behavior:**
1. First run: fetches N papers, adds M new entries (M > 0)
2. Second run on same query: `new_entries == 0` (all already marked known via SHA256)
3. SECOND-KNOWLEDGE-BRAIN.md table size is the same after second run

**Pass criteria:**
- `result_1["new_entries"] >= 1` ✓
- `result_2["new_entries"] == 0` ✓

---

## Scenario 8: All LLM Providers Unavailable — Graceful Degradation

**Setup:** Set ANTHROPIC_API_KEY, OPENAI_API_KEY to invalid values; set OLLAMA_BASE_URL to non-existent host.

**Expected behavior:**
1. LLM client falls back gracefully after 3 attempts per provider
2. SymptomAnalyzer uses heuristic OPQRST extraction
3. DiagnosisEngine uses ICD11_COMMON_PRESENTATIONS heuristic mapping
4. ReportGenerator uses template guidance
5. Full result returned (not an error crash)
6. `result["differentials"]` non-empty
7. Safety disclaimer still present in patient report

**Pass criteria:**
- No unhandled exception ✓
- `len(result["differentials"]) >= 2` ✓
- Disclaimer in patient report ✓
- `result["triage"]["severity"]` is one of LOW/MEDIUM/HIGH/CRITICAL ✓

---

## Scenario 9: REST API Integration

**Setup:** Start FastAPI server on port 8008. Send POST to `/api/v1/diagnose`.

**Request:**
```json
{
  "symptoms": "severe chest pain radiating to jaw, started 1 hour ago, 10/10, diaphoretic",
  "vitals": {"respiration_rate": 24, "pulse": 110, "systolic_bp": 100}
}
```

**Expected response structure:**
```json
{
  "session_id": "uuid-string",
  "triage": {
    "severity": "CRITICAL",
    "news2_score": ...,
    "escalation_required": true,
    "recommended_action": "CALL EMERGENCY..."
  },
  "differentials": [...],
  "patient_report": "Call emergency services...",
  "safety_alerts": [...],
  "safety_compliant": true
}
```

**Pass criteria:**
- HTTP 200 OK ✓
- `response["triage"]["severity"] == "CRITICAL"` ✓
- `response["triage"]["escalation_required"] == True` ✓
- `"session_id"` in response ✓
- `len(response["differentials"]) >= 1` ✓
