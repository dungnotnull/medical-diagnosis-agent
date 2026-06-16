# PROJECT-detail.md — Medical Diagnosis Agent

## Executive Summary

The **medical-diagnosis-agent** is an AI-powered clinical decision support system that provides evidence-based symptom assessment, validated severity triage, and differential diagnosis generation. It targets patients needing initial assessment, caregivers supporting loved ones, and clinicians seeking a fast decision-support layer. The agent conducts structured OPQRST symptom interviews, applies NEWS2/qSOFA/CURB-65/GCS scoring, maps presentations to ICD-11 differential diagnoses, and generates patient-safe, professionally formatted reports.

**Critical design constraint**: this agent is informational only. It never prescribes medication dosages. For any HIGH or CRITICAL triage score, it mandates immediate specialist/emergency consultation.

---

## Problem Statement

- 4.5 billion people lack access to essential health services (WHO 2023)
- Emergency department overcrowding: 30–50% of visits are non-urgent (AHRQ data)
- Rural and low-resource settings lack real-time clinical decision support
- Patients often misinterpret symptom severity, delaying or over-utilizing care

**Gap**: An AI agent that applies validated clinical frameworks to provide rapid, evidence-based triage and guidance — augmenting, not replacing, medical professionals.

---

## Target Users & Use Cases

| User | Trigger | Agent Action |
|------|---------|-------------|
| Patient | Describes symptoms via CLI/API | OPQRST interview → triage score → differential → patient-safe guidance |
| Caregiver | Describes elderly relative's symptoms | Severity assessment → when to call 911 vs. schedule appointment |
| Rural nurse | Quick clinical decision support | NEWS2 scoring → differential → evidence-cited recommendations |
| Emergency physician | Rapid pre-screen context | Structured symptom summary + ICD candidates for fast intake |

---

## Agent Architecture

```
User Input (text symptoms / structured form)
        ↓
┌─────────────────────────────────────────────────────────────────┐
│  MedicalDiagnosisOrchestrator (agent/orchestrator.py)           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ SymptomAnalyzer  │→ │  TriageEngine    │→ │DiagnosisEng. │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│           ↓                    ↓                    ↓           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           ReportGenerator (safety gates)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │    MemoryManager (SQLite — session audit trail)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        ↓              ↓                    ↓
   LLM API       HuggingFace          Medical APIs
(llm_client)   (hf_model_mgr)    (PubMed/WHO/NICE)
        ↓
  Patient Report + Doctor Summary + Safety Alerts
```

---

## Full Module Catalog

### `agent/modules/symptom_analyzer.py`
**Responsibility:** Extract, structure, and classify symptoms from free-text input.  
**Inputs:** Raw patient text, optional audio transcript  
**Outputs:** `SymptomProfile` dataclass (OPQRST fields, entity list, severity keywords, system affected)  
**Tools called:** `hf_model_manager.extract_medical_entities()`, LLM OPQRST interview prompt  
**Quality gate:** Minimum 2 OPQRST fields populated; at least 1 symptom entity extracted

### `agent/modules/triage_engine.py`
**Responsibility:** Apply validated clinical scoring systems and detect red-flag presentations.  
**Inputs:** `SymptomProfile`, vital signs (if provided)  
**Outputs:** `TriageResult` dataclass (NEWS2 score, qSOFA score, severity level LOW/MEDIUM/HIGH/CRITICAL, red_flags list, escalation_required bool)  
**Tools called:** NEWS2 calculator, qSOFA calculator, CURB-65 calculator, GCS calculator  
**Quality gate:** Escalation triggered for any red-flag; CRITICAL always → emergency referral

### `agent/modules/diagnosis_engine.py`
**Responsibility:** Generate ICD-11 differential diagnoses ranked by probability.  
**Inputs:** `SymptomProfile`, `TriageResult`  
**Outputs:** `DiagnosisResult` dataclass (differential list with ICD codes, confidence scores, evidence citations)  
**Tools called:** `hf_model_manager.classify_clinical_text()`, BGE-large retrieval from SECOND-KNOWLEDGE-BRAIN.md, LLM synthesis  
**Quality gate:** ≥2 differential candidates; all entries cite at least 1 evidence source

### `agent/modules/report_generator.py`
**Responsibility:** Assemble final patient-safe and doctor-summary reports with safety gates.  
**Inputs:** `SymptomProfile`, `TriageResult`, `DiagnosisResult`  
**Outputs:** Markdown patient report, doctor summary JSON, safety alerts  
**Tools called:** LLM patient guidance prompt, BART-CNN for evidence summarization  
**Quality gate:** No medication dosages in output; disclaimer present; escalation notice for HIGH/CRITICAL

---

## HuggingFace Model Selection

| Model | Task | Benchmark | Reason over alternatives |
|-------|------|-----------|--------------------------|
| `emilyalsentzer/Bio_ClinicalBERT` | Medical NER | F1=0.87 on i2b2-2010 | Trained on MIMIC-III clinical notes; best for EMR-style text |
| `NLP4Science/pubmedbert-full-text-clinical` | Text classification | AUROC=0.89 clinical NLP | PubMedBERT architecture; fine-tuned on full clinical texts |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity | MTEB avg 56.3 | 10× faster than BGE-large; fine for symptom lookup |
| `BAAI/bge-large-en-v1.5` | Dense retrieval | MTEB avg 63.6 (SOTA) | Best retrieval for knowledge base evidence lookup |
| `BAAI/bge-reranker-large` | Cross-encoder rerank | BEIR avg +8% over bi-encoder | Maximizes precision of top-3 evidence sources |
| `facebook/bart-large-cnn` | Summarization | ROUGE-L 40.9 on CNN/DM | Best open summarizer for abstractive PubMed condensation |

---

## LLM API Integration Spec

### Primary: Claude (`claude-opus-4-8`)

| Use Case | Prompt Template | Token Budget |
|----------|----------------|-------------|
| OPQRST interview | `OPQRST_INTERVIEW_PROMPT` | ~1,500 input / 800 output |
| Differential synthesis | `DIFFERENTIAL_SYNTHESIS_PROMPT` | ~3,000 input / 1,200 output |
| Patient guidance | `PATIENT_GUIDANCE_PROMPT` | ~2,000 input / 600 output |
| Evidence summary | `EVIDENCE_SYNTHESIS_PROMPT` | ~4,000 input / 800 output |

### Fallback chain: `gpt-4o` → `llama3` (Ollama)

---

## E2E Execution Flow

```
1. Receive user input (text symptoms or JSON structured form)
2. SymptomAnalyzer.analyze()
   a. Run Bio_ClinicalBERT NER → extract entities (symptoms, anatomy, duration)
   b. Send to LLM with OPQRST_INTERVIEW_PROMPT → structured OPQRST fields
   c. Classify body system affected (cardiovascular, respiratory, neurological, etc.)
   d. Detect severity keywords (severe, crushing, can't breathe, unconscious, etc.)
   → SymptomProfile
3. TriageEngine.score()
   a. Compute NEWS2 (if vitals available) else compute partial NEWS2
   b. Compute qSOFA (if applicable: suspected infection)
   c. Compute CURB-65 (if respiratory symptoms)
   d. Compute GCS (if altered consciousness reported)
   e. Detect red-flag patterns (FAST criteria, chest pain + diaphoresis, etc.)
   f. Assign overall severity: CRITICAL > HIGH > MEDIUM > LOW
   → TriageResult
4. SAFETY GATE: if CRITICAL or red_flags → generate immediate emergency alert, STOP normal flow
5. DiagnosisEngine.diagnose()
   a. Retrieve top-10 relevant papers from SECOND-KNOWLEDGE-BRAIN via BGE-large FAISS
   b. Rerank with BGE-reranker → top-5 evidence items
   c. Run PubMedBERT classification → ICD category probabilities
   d. Send to LLM with DIFFERENTIAL_SYNTHESIS_PROMPT → ranked differential list
   → DiagnosisResult
6. ReportGenerator.generate()
   a. Summarize evidence with BART-CNN
   b. Generate patient guidance with LLM (PATIENT_GUIDANCE_PROMPT)
   c. Generate doctor summary (structured JSON)
   d. Apply safety gates: strip any medication dosages, add disclaimers
   e. Add specialist referral notice if HIGH/CRITICAL
   → Patient report (Markdown) + Doctor summary (JSON)
7. MemoryManager.save_session()
8. Return final output
```

---

## SECOND-KNOWLEDGE-BRAIN Integration

- **Sources:** PubMed Central (NCBI Entrez API), Cochrane Library HTML, WHO guidelines, NICE guidelines, MedRxiv, ArXiv cs.AI+cs.LG (medical AI)
- **Update frequency:** Weekly (Sunday 02:00) via APScheduler
- **Crawl pipeline:** `tools/knowledge_updater.py` → fetch → score (recency×relevance) → deduplicate (SHA256) → append to `SECOND-KNOWLEDGE-BRAIN.md`
- **Retrieval:** BGE-large-en-v1.5 FAISS IndexFlatIP at query time

---

## Quality Gates

1. **Symptom extraction**: ≥2 OPQRST fields + ≥1 NER entity (else request clarification)
2. **Triage completeness**: NEWS2 or qSOFA score computed (not zero-defaulted silently)
3. **Differential coverage**: ≥2 ICD candidates with evidence citations
4. **Safety compliance**: output scanned for medication dosage patterns (regex) → blocked if found
5. **Disclaimer presence**: all patient-facing outputs must contain the safety disclaimer
6. **Escalation enforcement**: CRITICAL/HIGH triage → specialist referral notice mandatory
7. **Source citation**: all diagnostic statements must cite ≥1 paper or guideline

---

## Test Scenarios (see `tests/test-scenarios.md`)

1. Golden path: chest pain + shortness of breath → CRITICAL triage → emergency alert
2. Low-acuity: mild sore throat 2 days → LOW triage → self-care guidance + safety disclaimer
3. Neurological: sudden severe headache + neck stiffness → HIGH → FAST stroke check → specialist referral
4. Respiratory: productive cough + fever 3 days + age 70 → CURB-65 scoring → pneumonia differential
5. GCS: altered consciousness reported → GCS estimation → CRITICAL if <13
6. Knowledge crawl: PubMed dedup test — same PMID submitted twice → only 1 entry added
7. All LLM providers unavailable → graceful degradation to heuristic rules + SECOND-KNOWLEDGE-BRAIN
8. REST API: POST /api/v1/diagnose → structured JSON response with all fields

---

## Key Design Decisions

1. **NEWS2 as primary triage tool**: standardized by NHS; most widely validated early warning score globally
2. **ICD-11 over ICD-10**: WHO released ICD-11 in 2022; better AI-parseable structure
3. **Bio_ClinicalBERT over general BERT**: clinical notes training makes it 15–20% better on medical NER
4. **No medication dosages ever**: agent cannot know patient's contraindications, allergies, or kidney/liver function
5. **OPQRST interview structure**: gold-standard emergency medicine symptom interview framework
6. **Safety-first output pipeline**: ReportGenerator applies safety checks LAST, after LLM generation — cannot be bypassed
7. **SQLite with WAL mode**: sufficient for single-instance clinical sessions; upgrade to PostgreSQL if multi-user deployment needed
