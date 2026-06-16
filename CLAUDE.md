# CLAUDE.md — Medical Diagnosis Agent (Folder 8)

## Agent Identity
**Name:** medical-diagnosis-agent  
**Tagline:** Evidence-based clinical triage, differential diagnosis, and emergency support AI agent  
**Build Phase:** Phase 1 — Core Implementation  
**Cluster:** F — Domain Intelligence & Analytics Agents

---

## Problem Statement

Millions of people worldwide face delayed or inaccessible medical assessment — especially in emergencies, rural areas, or during off-hours. This agent provides an AI-powered clinical decision support system that conducts structured symptom interviews (OPQRST methodology), applies validated medical scoring systems (NEWS2, qSOFA, CURB-65, GCS), maps symptoms to ICD-11 differential diagnoses, and generates patient-safe reports. For serious or life-threatening presentations, the agent always escalates to specialist consultation. It never recommends specific medication dosages. Its medical knowledge base is continuously updated from PubMed Central, Cochrane Library, WHO guidelines, NICE guidelines, and MedRxiv preprints.

---

## Agent Architecture (10-Step Pipeline)

```
User Symptom Input (text/structured)
        ↓
1. SymptomAnalyzer    — OPQRST extraction, Bio_ClinicalBERT NER, severity keywords
        ↓
2. TriageEngine       — NEWS2/qSOFA/CURB-65/GCS scoring, red-flag detection
        ↓
3. DiagnosisEngine    — ICD-11 mapping, PubMedBERT classification, LLM differential
        ↓
4. ReportGenerator    — Patient report, doctor summary, safety gates
        ↓
5. MemoryManager      — Persist session, audit trail, cost tracking
        ↓
Output: Triage severity + Differential diagnoses + Patient-safe guidance
```

---

## Module List (`agent/modules/`)

| File | Description |
|------|-------------|
| `symptom_analyzer.py` | OPQRST interview, Bio_ClinicalBERT NER, symptom cluster extraction |
| `triage_engine.py` | NEWS2/qSOFA/CURB-65/GCS scoring, red-flag detection, escalation triggers |
| `diagnosis_engine.py` | ICD-11 differential mapping, PubMedBERT classification, LLM synthesis |
| `report_generator.py` | Patient-safe guidance, doctor summary, safety gate enforcement |

---

## Tools (`agent/tools/` — inside `tools/`)

| File | Description |
|------|-------------|
| `knowledge_updater.py` | Crawl PubMed, Cochrane, WHO, NICE, MedRxiv → SECOND-KNOWLEDGE-BRAIN.md |
| `llm_client.py` | Claude primary / OpenAI fallback / Ollama offline LLM client |
| `hf_model_manager.py` | HuggingFace model registry: Bio_ClinicalBERT, PubMedBERT, MiniLM |

---

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|-----------|
| `emilyalsentzer/Bio_ClinicalBERT` | Medical NER (symptom/anatomy extraction) | Pre-trained on MIMIC clinical notes; strong medical entity recognition |
| `NLP4Science/pubmedbert-full-text-clinical` | Clinical text classification | PubMedBERT fine-tuned on clinical data; ICD category relevance scoring |
| `sentence-transformers/all-MiniLM-L6-v2` | Symptom semantic search | Lightweight, fast cosine similarity for symptom→ICD mapping |
| `BAAI/bge-large-en-v1.5` | Knowledge base retrieval | Highest MTEB retrieval score; powers evidence retrieval from knowledge brain |
| `BAAI/bge-reranker-large` | Evidence reranking | Cross-encoder reranker for clinical evidence relevance |
| `facebook/bart-large-cnn` | Evidence summarization | Condensing PubMed abstracts for LLM context |

---

## LLM API Integration

| Provider | Priority | Use Cases |
|----------|----------|-----------|
| `claude-opus-4-8` | Primary | OPQRST structured interview, differential diagnosis synthesis, patient guidance |
| `gpt-4o` | Fallback | Same tasks when Claude API unavailable |
| `llama3` (Ollama) | Offline | Privacy mode — no PHI sent to cloud; local inference only |

---

## Knowledge Crawl Sources

| Source | Type | Frequency |
|--------|------|-----------|
| PubMed Central (NCBI Entrez API) | Clinical research papers | Weekly |
| Cochrane Library | Systematic reviews | Weekly |
| WHO Clinical Guidelines | Official guidelines | Weekly |
| NICE guidelines (UK) | Evidence-based clinical guidelines | Weekly |
| MedRxiv preprints | Latest medical research | Weekly |
| ArXiv cs.AI + cs.LG | AI/ML medical applications | Weekly |

---

## Safety Gates (CRITICAL)

1. **Red-flag escalation**: chest pain + diaphoresis, stroke FAST criteria, respiratory distress → ALWAYS escalate to emergency services
2. **No medication dosages**: agent NEVER recommends specific drug doses
3. **Always defer for serious cases**: any HIGH/CRITICAL triage score → mandatory specialist referral
4. **Disclaimer on all outputs**: all recommendations are informational only, not a substitute for medical care
5. **PRIVACY_MODE support**: Ollama offline mode for sensitive patient data

---

## Active Development Tasks

- [x] CLAUDE.md
- [x] PROJECT-detail.md
- [x] PROJECT-DEVELOPMENT-PHASE-TRACKING.md
- [x] SECOND-KNOWLEDGE-BRAIN.md
- [x] agent/main.py
- [x] agent/orchestrator.py
- [x] agent/modules/symptom_analyzer.py
- [x] agent/modules/triage_engine.py
- [x] agent/modules/diagnosis_engine.py
- [x] agent/modules/report_generator.py
- [x] agent/memory/memory_manager.py
- [x] tools/knowledge_updater.py
- [x] tools/llm_client.py
- [x] tools/hf_model_manager.py
- [x] config/agent_config.yaml
- [x] config/.env.example
- [x] docker/docker-compose.yml
- [x] tests/test-scenarios.md
- [x] tests/test_agent.py
