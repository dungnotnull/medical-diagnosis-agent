"""Medical Diagnosis Agent — CLI entry point and FastAPI server."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("medical-diagnosis-agent")


def _load_config() -> dict:
    config_path = Path("config/agent_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@click.group()
def cli():
    """Medical Diagnosis Agent — evidence-based clinical triage and differential diagnosis."""


@cli.command()
@click.argument("symptoms", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read symptoms from file")
@click.option("--vitals", "-v", type=str, help="JSON string of vital signs")
@click.option("--output", "-o", type=click.Choice(["text", "json", "markdown"]), default="text")
def diagnose(symptoms, file, vitals, output):
    """Run a clinical assessment for given symptoms."""
    if file:
        with open(file) as fh:
            patient_text = fh.read().strip()
    elif symptoms:
        patient_text = symptoms
    else:
        click.echo("Enter patient symptoms (Ctrl+D when done):")
        patient_text = sys.stdin.read().strip()

    if not patient_text:
        click.echo("Error: no symptom text provided.", err=True)
        sys.exit(1)

    vitals_dict = None
    if vitals:
        try:
            vitals_dict = json.loads(vitals)
        except json.JSONDecodeError:
            click.echo("Warning: could not parse vitals JSON — proceeding without vitals", err=True)

    config = _load_config()
    from agent.orchestrator import MedicalDiagnosisOrchestrator
    orchestrator = MedicalDiagnosisOrchestrator(config)

    click.echo("Analyzing symptoms...", err=True)
    result = orchestrator.diagnose_sync(patient_text, vitals_dict)

    if output == "json":
        click.echo(json.dumps(result, indent=2))
    elif output == "markdown":
        _print_markdown(result)
    else:
        _print_text(result)


def _print_text(result: dict) -> None:
    triage = result["triage"]
    click.echo(f"\n{'='*60}")
    click.echo(f"TRIAGE SEVERITY: {triage['severity']}")
    click.echo(f"NEWS2 Score: {triage['news2_score']} | qSOFA: {triage['qsofa_score']} | CURB-65: {triage['curb65_score']}")
    if triage["red_flags"]:
        click.echo(f"\n⚠️  RED FLAGS:")
        for rf in triage["red_flags"]:
            click.echo(f"  • {rf}")
    click.echo(f"\nRECOMMENDED ACTION: {triage['recommended_action']}")
    click.echo(f"\n{'='*60}")
    click.echo("DIFFERENTIAL DIAGNOSES:")
    for i, d in enumerate(result["differentials"], 1):
        urgent_tag = " [URGENT]" if d["urgent"] else ""
        click.echo(f"  {i}. {d['condition_name']} ({d['icd_code']}) — {d['probability']} probability{urgent_tag}")
    click.echo(f"\n{'='*60}")
    click.echo("PATIENT GUIDANCE:")
    click.echo(result["patient_report"])
    if result["safety_alerts"]:
        click.echo(f"\n⚠️  SAFETY ALERTS:")
        for alert in result["safety_alerts"]:
            click.echo(f"  {alert}")


def _print_markdown(result: dict) -> None:
    triage = result["triage"]
    lines = [
        f"# Medical Assessment — Session {result['session_id'][:8]}",
        f"",
        f"## Triage Result: **{triage['severity']}**",
        f"",
        f"| Score | Value |",
        f"|-------|-------|",
        f"| NEWS2 | {triage['news2_score']} |",
        f"| qSOFA | {triage['qsofa_score']} |",
        f"| CURB-65 | {triage['curb65_score']} |",
        f"",
        f"**Recommended Action:** {triage['recommended_action']}",
        f"",
        f"## Differential Diagnoses",
        f"",
    ]
    for i, d in enumerate(result["differentials"], 1):
        lines.append(f"{i}. **{d['condition_name']}** (`{d['icd_code']}`) — {d['probability']} probability")
        if d["evidence_citations"]:
            lines.append(f"   *Evidence: {', '.join(d['evidence_citations'][:2])}*")
    lines.extend([f"", f"## Patient Guidance", f"", result["patient_report"]])
    click.echo("\n".join(lines))


@cli.command("update-knowledge")
def update_knowledge():
    """Crawl PubMed, Cochrane, WHO, MedRxiv and update SECOND-KNOWLEDGE-BRAIN.md."""
    config = _load_config()
    from agent.orchestrator import MedicalDiagnosisOrchestrator
    orchestrator = MedicalDiagnosisOrchestrator(config)
    click.echo("Running knowledge update...")
    result = asyncio.run(orchestrator.update_knowledge())
    click.echo(f"Knowledge update complete: {result.get('new_entries', 0)} new entries added")


@cli.command("cost-report")
def cost_report():
    """Show LLM API cost summary for the last 30 days."""
    config = _load_config()
    from agent.orchestrator import MedicalDiagnosisOrchestrator
    orchestrator = MedicalDiagnosisOrchestrator(config)
    report = orchestrator.get_cost_report()
    click.echo(json.dumps(report, indent=2))


@cli.command()
def stats():
    """Show agent usage statistics."""
    config = _load_config()
    from agent.orchestrator import MedicalDiagnosisOrchestrator
    orchestrator = MedicalDiagnosisOrchestrator(config)
    click.echo(json.dumps(orchestrator.get_stats(), indent=2))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8008, help="Server port")
@click.option("--start-scheduler", is_flag=True, default=False, help="Start weekly knowledge update scheduler")
def serve(host, port, start_scheduler):
    """Start the FastAPI REST server."""
    import uvicorn
    from agent.main import create_app
    config = _load_config()
    app = create_app(config, start_scheduler=start_scheduler)
    uvicorn.run(app, host=host, port=port, log_level="info")


def create_app(config: dict, start_scheduler: bool = False):
    """Create and return the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse
        from pydantic import BaseModel
        from typing import Optional
    except ImportError:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from agent.orchestrator import MedicalDiagnosisOrchestrator

    app = FastAPI(
        title="Medical Diagnosis Agent",
        description="Evidence-based clinical triage and differential diagnosis AI",
        version="1.0.0",
    )
    orchestrator = MedicalDiagnosisOrchestrator(config)
    if start_scheduler:
        orchestrator.start_scheduler()

    class DiagnoseRequest(BaseModel):
        symptoms: str
        vitals: Optional[dict] = None

    class KnowledgeUpdateResponse(BaseModel):
        new_entries: int
        message: str

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "medical-diagnosis-agent"}

    @app.post("/api/v1/diagnose")
    async def diagnose_endpoint(req: DiagnoseRequest):
        if not req.symptoms or len(req.symptoms.strip()) < 5:
            raise HTTPException(status_code=422, detail="Symptom text too short")
        result = await orchestrator.diagnose(req.symptoms, req.vitals)
        return result

    @app.get("/api/v1/sessions")
    def list_sessions(limit: int = 20):
        return orchestrator._get_memory().get_recent_sessions(limit)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str):
        session = orchestrator._get_memory().get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.post("/api/v1/knowledge/update")
    async def knowledge_update():
        result = await orchestrator.update_knowledge()
        return KnowledgeUpdateResponse(
            new_entries=result.get("new_entries", 0),
            message=result.get("message", "Update complete"),
        )

    @app.get("/api/v1/cost")
    def cost():
        return orchestrator.get_cost_report()

    @app.get("/api/v1/stats")
    def agent_stats():
        return orchestrator.get_stats()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return orchestrator.get_prometheus_metrics()

    return app


if __name__ == "__main__":
    cli()
