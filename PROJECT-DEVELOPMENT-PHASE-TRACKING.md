# PROJECT-DEVELOPMENT-PHASE-TRACKING.md — Medical Diagnosis Agent

## Quantified Improvement Targets

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Triage accuracy (NEWS2) | Manual calculation | ≥92% agreement with clinical benchmark | Against Beth Israel Deaconess Hospital dataset |
| Differential diagnosis top-3 accuracy | GPT-3.5 baseline ~55% | ≥70% top-3 hit rate | Human physician evaluation on 50 cases |
| Red-flag detection sensitivity | No baseline | ≥99% sensitivity (zero miss rate) | Against RCEM red-flag criteria checklist |
| Knowledge base growth | 0 entries | ≥15 papers/week | Automated count from weekly crawl |
| Patient report safety compliance | N/A | 100% — zero medication dosages in output | Regex scan of all generated reports |

---

## Phase 0: Research & Architecture (Week 1–2)

### Tasks
- [x] Read ICD-11 linearization documentation and API spec
- [x] Survey validated clinical scoring systems: NEWS2, qSOFA, CURB-65, GCS, FAST
- [x] Evaluate HuggingFace models: Bio_ClinicalBERT, PubMedBERT, MiniLM for clinical NLP
- [x] Define OPQRST interview prompt templates
- [x] Design safety gate architecture (must be at LLM output post-processing stage)
- [x] Design SQLite schema for session audit trail

### Deliverables
- Architecture diagram (see PROJECT-detail.md)
- Safety gate specification
- HuggingFace model selection rationale

### Success Criteria
- All 5 scoring systems documented with input/output specs
- Safety gates cannot be bypassed by LLM output format
- Model selection validated against clinical NLP benchmarks

**Estimated effort:** 5 person-days

---

## Phase 1: Core Agent Modules (Week 3–5)

### Tasks
- [x] `symptom_analyzer.py`: OPQRST extraction + Bio_ClinicalBERT NER
- [x] `triage_engine.py`: NEWS2/qSOFA/CURB-65/GCS calculators + red-flag detection
- [x] `diagnosis_engine.py`: ICD-11 differential mapping + LLM synthesis
- [x] `report_generator.py`: patient report + doctor summary + safety gates
- [x] `agent/memory/memory_manager.py`: SQLite WAL session storage

### Deliverables
- 4 module files with full implementation
- Memory manager with session audit trail

### Success Criteria
- TriageEngine: NEWS2 computes correctly for all 6 physiological parameters
- Red-flag detection: FAST criteria, chest pain + diaphoresis, respiratory distress all trigger CRITICAL
- DiagnosisEngine: returns ≥2 ICD candidates with confidence scores
- ReportGenerator: regex scan confirms no medication dosages in output

**Estimated effort:** 8 person-days

---

## Phase 2: Orchestrator + Quality Gates (Week 6–8)

### Tasks
- [x] `agent/orchestrator.py`: 10-step E2E pipeline with async orchestration
- [x] Wire modules: SymptomAnalyzer → TriageEngine → DiagnosisEngine → ReportGenerator
- [x] Implement safety gate enforcement at orchestrator level
- [x] Add APScheduler for weekly knowledge updates (Sunday 02:00)
- [x] Prometheus metrics exposition (session_count, triage_level_counts, llm_cost_total)

### Deliverables
- Orchestrator with full E2E pipeline
- APScheduler integration
- Prometheus metrics

### Success Criteria
- E2E pipeline completes in <10 seconds for simple cases
- Safety gates trigger correctly for all CRITICAL/HIGH presentations
- Knowledge update runs without errors on APScheduler trigger

**Estimated effort:** 6 person-days

---

## Phase 3: HuggingFace Model Integration (Week 9–10)

### Tasks
- [x] `tools/hf_model_manager.py`: Bio_ClinicalBERT, PubMedBERT, MiniLM, BGE-large, BGE-reranker, BART-CNN
- [x] Lazy loading: models download on first use, cache in `./models/`
- [x] CUDA auto-detection; CPU fallback if no GPU
- [x] Idle timeout: unload model after 600 seconds of inactivity
- [x] TF-IDF fallback if HuggingFace Hub unavailable

### Deliverables
- `hf_model_manager.py` with full registry
- Fallback mechanisms for offline operation

### Success Criteria
- Bio_ClinicalBERT extracts ≥5 entities from standard chest pain presentation
- BGE-large retrieval returns relevant papers for "chest pain differential diagnosis"
- BART summarization produces ≤150-word summary of PubMed abstract

**Estimated effort:** 5 person-days

---

## Phase 4: LLM API Integration (Week 11–12)

### Tasks
- [x] `tools/llm_client.py`: Claude/OpenAI/Ollama chain with exponential backoff
- [x] OPQRST_INTERVIEW_PROMPT: structured JSON output with all 6 OPQRST fields
- [x] DIFFERENTIAL_SYNTHESIS_PROMPT: ranked ICD list with confidence + evidence citations
- [x] PATIENT_GUIDANCE_PROMPT: plain-language guidance (no jargon, no dosages)
- [x] EVIDENCE_SYNTHESIS_PROMPT: clinical evidence summarization
- [x] PRIVACY_MODE: force Ollama for all PHI-containing sessions

### Deliverables
- `llm_client.py` with streaming support
- 4 validated prompt templates

### Success Criteria
- All 4 prompts produce valid JSON output on first attempt ≥95% of the time
- Fallback chain: Claude → OpenAI → Ollama works correctly
- PRIVACY_MODE correctly bypasses cloud APIs

**Estimated effort:** 5 person-days

---

## Phase 5: SECOND-KNOWLEDGE-BRAIN Pipeline (Week 13–14)

### Tasks
- [x] `tools/knowledge_updater.py`: PubMed (NCBI Entrez) + Cochrane + WHO + MedRxiv crawl pipeline
- [x] Score papers: 0.6×recency + 0.4×relevance (medical domain keywords)
- [x] Deduplicate via PMID/DOI hash
- [x] Append to SECOND-KNOWLEDGE-BRAIN.md (table format)
- [x] Run initial seed crawl: 15+ foundation papers

### Deliverables
- `knowledge_updater.py` with 5 medical sources
- `SECOND-KNOWLEDGE-BRAIN.md` with initial 15 papers

### Success Criteria
- First crawl adds ≥15 papers without duplicates
- Second crawl of same query adds 0 duplicate papers
- Weekly APScheduler trigger runs at Sunday 02:00

**Estimated effort:** 5 person-days

---

## Phase 6: Docker + Testing (Week 15–16)

### Tasks
- [x] `docker/docker-compose.yml`: medical-diagnosis-agent + ollama (optional profile)
- [x] `docker/Dockerfile`: python:3.12-slim, non-root user, EXPOSE 8008
- [x] `tests/test-scenarios.md`: 8 end-to-end test scenarios
- [x] `tests/test_agent.py`: 40+ automated unit/integration tests
- [x] `config/agent_config.yaml` + `config/.env.example`

### Deliverables
- Docker deployment stack
- Full test suite

### Success Criteria
- All 8 test scenarios pass
- Docker container starts and responds to `GET /health` within 30 seconds
- No medication dosages in any test output (automated check)

**Estimated effort:** 6 person-days

---

## Phase 7: Cross-Agent Wiring & Deployment (Week 17–18)

### Tasks
- [x] Integrate with `ai-benchmark-agent` (folder 22) for LLM performance tracking
- [x] Integrate with `academic-research-enhanced` (folder 18) for additional paper discovery
- [x] Enable Prometheus metrics scraping from `dockprom-enhanced` (folder 14)
- [x] Production hardening: rate limiting, PHI scrubbing in logs, audit trail encryption

### Deliverables
- Cross-agent integration documentation
- Production deployment checklist

### Success Criteria
- Benchmark agent records latency + cost for all diagnosis sessions
- Academic agent can push relevant medical papers to knowledge brain
- No PHI in Prometheus metrics labels

**Estimated effort:** 4 person-days

---

## Total Estimated Effort: 44 person-days

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Bio_ClinicalBERT returns low-confidence NER | Medium | Medium | Fallback to LLM NER extraction |
| PubMed Entrez API rate limit (3 req/s) | High | Low | Throttle to 2 req/s + retry |
| LLM generates medication dosages despite instructions | Low | Critical | Post-processing regex gate blocks output |
| User treats agent output as definitive medical advice | Medium | Critical | Prominent disclaimer on all outputs; CRITICAL always escalates |
| HuggingFace Hub unavailable | Low | Medium | TF-IDF fallback for all embedding tasks |
