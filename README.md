# Medical Diagnosis Agent

Evidence-based clinical triage, differential diagnosis, and emergency support AI agent.

## Overview

The Medical Diagnosis Agent is an AI-powered clinical decision support system that conducts structured symptom interviews (OPQRST methodology), applies validated medical scoring systems (NEWS2, qSOFA, CURB-65, GCS), maps symptoms to ICD-11 differential diagnoses, and generates patient-safe reports. For serious or life-threatening presentations, the agent always escalates to specialist consultation.

## ⚠️ Medical Disclaimer

**This agent is for informational purposes only and does NOT replace medical advice, diagnosis, or treatment from a qualified healthcare provider.** If you are experiencing a medical emergency, call emergency services (911/999) immediately.

## Features

- **OPQRST Symptom Interview**: Structured clinical assessment using Onset, Provocation, Quality, Radiation, Severity, Time methodology
- **Validated Scoring Systems**: NEWS2, qSOFA, CURB-65, GCS, FAST stroke criteria
- **ICD-11 Differential Diagnosis**: Evidence-based condition mapping with confidence scores
- **Red-Flag Detection**: Automatic escalation for life-threatening presentations
- **Safety Gates**: Regex-based blocking of medication dosages in all outputs
- **Privacy Mode**: Local Ollama support for offline PHI processing
- **Knowledge Crawler**: Weekly automated updates from PubMed, Cochrane, WHO, MedRxiv
- **Cross-Agent Integration**: Benchmark tracking, academic research integration, Prometheus metrics

## Quick Start

### Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/medical-diagnosis-agent.git
cd medical-diagnosis-agent

# Configure environment
cp config/.env.example config/.env
# Edit config/.env with your API keys

# Start the agent
docker-compose up -d

# Run a diagnosis
curl -X POST http://localhost:8008/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "mild headache and fatigue for 2 days"}'
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Run the server
python -m agent.main serve --host 0.0.0.0 --port 8008

# Or use the CLI
python -m agent.main diagnose "chest pain since 1 hour" --output markdown
```

## Architecture

```
User Symptom Input (text/structured)
        ↓
1. SymptomAnalyzer    — OPQRST extraction, Bio_ClinicalBERT NER, severity keywords
        ↓
2. TriageEngine       — NEWS2/qSOFA/CURB-65/GCS scoring, red-flag detection
        ↓
3. DiagnosisEngine    — ICD-11 differential mapping, PubMedBERT classification, LLM synthesis
        ↓
4. ReportGenerator    — Patient report, doctor summary, safety gates
        ↓
5. MemoryManager      — Persist session, audit trail, cost tracking
        ↓
Output: Triage severity + Differential diagnoses + Patient-safe guidance
```

## Modules

| Module | Description |
|--------|-------------|
| `symptom_analyzer.py` | OPQRST interview, Bio_ClinicalBERT NER, symptom cluster extraction |
| `triage_engine.py` | NEWS2/qSOFA/CURB-65/GCS scoring, red-flag detection, escalation triggers |
| `diagnosis_engine.py` | ICD-11 differential mapping, PubMedBERT classification, LLM synthesis |
| `report_generator.py` | Patient-safe guidance, doctor summary, safety gate enforcement |

## Tools

| Tool | Description |
|------|-------------|
| `knowledge_updater.py` | Crawl PubMed, Cochrane, WHO, NICE, MedRxiv → SECOND-KNOWLEDGE-BRAIN.md |
| `llm_client.py` | Claude primary / OpenAI fallback / Ollama offline LLM client |
| `hf_model_manager.py` | HuggingFace model registry: Bio_ClinicalBERT, PubMedBERT, MiniLM |

## HuggingFace Models

| Model ID | Task | Description |
|----------|------|-------------|
| `emilyalsentzer/Bio_ClinicalBERT` | Medical NER | Pre-trained on MIMIC clinical notes; symptom/anatomy extraction |
| `NLP4Science/pubmedbert-full-text-clinical` | Classification | PubMedBERT for ICD category relevance scoring |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic Search | Lightweight cosine similarity for symptom→ICD mapping |
| `BAAI/bge-large-en-v1.5` | Knowledge Retrieval | High-accuracy retrieval from knowledge brain |
| `BAAI/bge-reranker-large` | Evidence Reranking | Cross-encoder for clinical evidence relevance |
| `facebook/bart-large-cnn` | Summarization | Condensing PubMed abstracts for LLM context |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/diagnose` | POST | Run diagnosis (JSON: `{symptoms, vitals}`) |
| `/api/v1/sessions` | GET | List recent sessions |
| `/api/v1/sessions/{id}` | GET | Get session details |
| `/api/v1/knowledge/update` | POST | Trigger knowledge update |
| `/api/v1/cost` | GET | LLM cost summary (last 30 days) |
| `/api/v1/stats` | GET | Agent usage statistics |
| `/metrics` | GET | Prometheus metrics |

## Configuration

See `config/agent_config.yaml` for all configuration options:

```yaml
agent:
  name: medical-diagnosis-agent
  version: "1.0.0"

server:
  host: "0.0.0.0"
  port: 8008

llm:
  provider_chain: [claude, openai, ollama]
  default_claude_model: "claude-sonnet-4-6"
  privacy_mode: false

hf_models:
  idle_timeout_seconds: 600

triage_engine:
  scoring_systems: [news2, qsofa, curb65, gcs]

report_generator:
  safety_gates:
    block_medication_dosages: true
    require_disclaimer: true
```

## Safety Gates

The agent enforces multiple safety layers:

1. **Red-Flag Escalation**: Chest pain + diaphoresis, stroke FAST criteria, respiratory distress → ALWAYS escalate
2. **No Medication Dosages**: Regex blocks all dosage patterns in output
3. **Disclaimer on All Outputs**: Informational-only warning on every report
4. **Privacy Mode Support**: Ollama offline mode for sensitive patient data
5. **Critical Triage Handling**: Automatic emergency services recommendation

## Testing

```bash
# Run all tests
pytest tests/test_agent.py -v

# Run specific test
pytest tests/test_agent.py::TestTriageEngine::test_critical_chest_pain_diaphoresis -v

# Run with coverage
pytest --cov=agent tests/

# Test scenarios
cat tests/test-scenarios.md
```

## Development

### Project Status

All phases (0-7) are **100% complete**:

- ✅ Phase 0: Research & Architecture
- ✅ Phase 1: Core Agent Modules
- ✅ Phase 2: Orchestrator + Quality Gates
- ✅ Phase 3: HuggingFace Model Integration
- ✅ Phase 4: LLM API Integration
- ✅ Phase 5: SECOND-KNOWLEDGE-BRAIN Pipeline
- ✅ Phase 6: Docker + Testing
- ✅ Phase 7: Cross-Agent Wiring & Deployment

See `PROJECT-DEVELOPMENT-PHASE-TRACKING.md` for detailed progress.

### Code Quality

```bash
# Format code
ruff format agent/ tools/

# Type check
mypy agent/

# Lint
ruff check agent/ tools/
```

## Deployment

See `DEPLOYMENT.md` for comprehensive deployment instructions.

### Production Checklist

- [ ] All API keys configured in `.env`
- [ ] Rate limiting enabled and tested
- [ ] PHI scrubbing verified with test data
- [ ] Audit trail encryption configured
- [ ] Prometheus scrape target added
- [ ] Health check endpoint accessible
- [ ] Safety gates tested with medication dosage prompts
- [ ] Privacy mode tested with PHI data

## Monitoring

Key Prometheus metrics:

- `medical_diagnosis_sessions_total` — Total sessions
- `medical_triage_severity_counts` — Triage by severity level
- `medical_escalations_total` — Emergency escalations
- `medical_llm_cost_usd_total` — Cumulative LLM costs
- `medical_safety_gate_triggers_total` — Safety activations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

- Documentation: See `CLAUDE.md`, `PROJECT-detail.md`
- Issues: GitHub Issues
- Deployment: See `DEPLOYMENT.md`

## Acknowledgments

- ICD-11 classification by WHO
- Clinical scoring systems: NEWS2 (Royal College of Physicians), qSOFA (Sepsis-3), CURB-65 (British Thoracic Society), GCS (Teasdale & Jennett)
- HuggingFace models: Bio_ClinicalBERT, PubMedBERT, MiniLM, BGE, BART
- Medical knowledge sources: PubMed Central, Cochrane Library, WHO, NICE, MedRxiv

---

**Remember**: This agent is a decision support tool, not a replacement for professional medical care. Always use clinical judgment and follow local protocols.
