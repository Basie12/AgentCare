"""Workflow service — the seam between HTTP routes and the agent graph.

Responsibilities:
  * create a WorkflowRun and a stable thread_id
  * invoke the graph, catching interrupts
  * mirror the graph state into WorkflowRun.state so it is queryable in SQL
  * resume an interrupted run once a staff decision is persisted
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.agents.graph import (
    get_graph,
    graph_config,
    interrupt_payload,
    is_interrupted,
    resume_command,
)
from app.agents.state import initial_state
from app.db.models import (
    AuditEvent,
    Escalation,
    EscalationStatus,
    WorkflowRun,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

SERIALISABLE_KEYS = [
    "patient", "safety_verdict", "safety_rule", "blocked", "department", "routing_confidence",
    "completion",
    "routing_rationale", "routing_alternatives", "timeframe", "timeframe_note",
    "appointment_intent", "existing_appointments",
    "available_slots", "appointment",
    "documents", "duplicates", "missing_documents", "reminders", "requires_approval",
    "approval_reason", "approval_decision", "escalation_id", "current_step",
    "degraded", "errors", "agent_log", "final_message",
]


def _snapshot(state: dict) -> dict:
    return {k: state.get(k) for k in SERIALISABLE_KEYS if k in state}


def _audit(db: Session, actor_id: str | None, action: str, entity_id: str, meta: dict) -> None:
    db.add(
        AuditEvent(
            actor_id=actor_id,
            actor_type="system",
            action=action,
            entity_type="workflow_run",
            entity_id=entity_id,
            event_metadata=meta,
        )
    )


def start_workflow(
    db: Session,
    *,
    patient_id: str,
    actor_id: str,
    request_text: str,
    uploaded_files: list[dict] | None = None,
) -> WorkflowRun:
    thread_id = f"wf-{uuid.uuid4().hex[:16]}"
    run = WorkflowRun(
        thread_id=thread_id,
        patient_id=patient_id,
        request_text=request_text,
        current_step="created",
        status=WorkflowStatus.RUNNING,
        state={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    state = initial_state(
        workflow_run_id=run.id,
        thread_id=thread_id,
        patient_id=patient_id,
        actor_id=actor_id,
        request_text=request_text,
        uploaded_files=uploaded_files or [],
    )

    _run_graph(db, run, state)
    return run


def _run_graph(db: Session, run: WorkflowRun, payload: Any) -> WorkflowRun:
    graph = get_graph()
    config = graph_config(run.thread_id)

    try:
        result = graph.invoke(payload, config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("graph invocation failed for %s", run.id)
        run.status = WorkflowStatus.FAILED
        run.current_step = "error"
        run.state = {**(run.state or {}), "fatal_error": f"{type(exc).__name__}: {exc}"}
        _audit(db, None, "workflow.failed", run.id, {"error": str(exc)})
        db.commit()
        return run

    if is_interrupted(result):
        payload_data = interrupt_payload(result) or {}
        run.status = WorkflowStatus.AWAITING_APPROVAL
        run.current_step = "awaiting_approval"
        run.state = {**_snapshot(result), "interrupt": payload_data}
        _audit(db, None, "workflow.interrupted", run.id, {"reason": payload_data.get("reason")})
        db.commit()
        db.refresh(run)
        return run

    snapshot = _snapshot(result)
    run.state = snapshot
    run.current_step = snapshot.get("current_step", "completed")
    run.degraded = bool(snapshot.get("degraded"))
    run.status = (
        WorkflowStatus.BLOCKED
        if snapshot.get("blocked")
        else WorkflowStatus.COMPLETED
    )
    _audit(
        db,
        None,
        "workflow.completed",
        run.id,
        {"step": run.current_step, "status": run.status.value, "degraded": run.degraded},
    )
    db.commit()
    db.refresh(run)
    return run


def resolve_escalation(
    db: Session,
    *,
    escalation_id: str,
    reviewer_id: str,
    decision: str,
    note: str | None = None,
    department: str | None = None,
) -> WorkflowRun | None:
    """Persist the staff decision, then resume the paused graph from its checkpoint."""
    escalation = db.get(Escalation, escalation_id)
    if not escalation or escalation.status is not EscalationStatus.OPEN:
        return None

    escalation.status = (
        EscalationStatus.APPROVED if decision == "approved" else EscalationStatus.REJECTED
    )
    escalation.reviewed_by = reviewer_id
    escalation.reviewer_note = note

    run = db.get(WorkflowRun, escalation.workflow_run_id)
    if not run:
        db.commit()
        return None

    _audit(
        db,
        reviewer_id,
        f"escalation.{decision}",
        run.id,
        {"escalation_id": escalation_id, "note": note, "department": department},
    )
    db.commit()

    if run.status is not WorkflowStatus.AWAITING_APPROVAL:
        # Escalation was informational (e.g. emergency flag) — nothing to resume.
        return run

    run.status = WorkflowStatus.RUNNING
    db.commit()

    return _run_graph(db, run, resume_command(decision, department))
