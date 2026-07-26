"""The single typed state object passed between agents.

Each agent reads a defined slice and writes a defined slice — this is what
makes state transfer between agents explicit and inspectable rather than
implied.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict


def merge_list(left: list, right: list) -> list:
    """Reducer so parallel nodes append rather than overwrite."""
    return (left or []) + (right or [])


class AgentCareState(TypedDict, total=False):
    # --- identity / request ---
    workflow_run_id: str
    thread_id: str
    patient_id: str
    actor_id: str
    request_text: str
    uploaded_files: list[dict]  # [{filename, content_b64}]

    # --- safety agent output ---
    safety_verdict: Literal["proceed", "emergency", "clinical_advice", "sensitive"]
    safety_rule: str | None
    safety_message: str | None
    blocked: bool

    # --- routing agent output ---
    department: str | None
    routing_confidence: float
    routing_rationale: str
    routing_alternatives: list[str]

    # --- appointment agent output ---
    timeframe: dict | None
    timeframe_note: str
    available_slots: list[dict]
    selected_slot_id: str | None
    appointment: dict | None

    # --- document agent output ---
    documents: list[dict]
    duplicates: list[dict]
    missing_documents: list[str]

    # --- follow-up agent output ---
    reminders: list[dict]

    # --- oversight ---
    requires_approval: bool
    approval_reason: str | None
    approval_decision: Literal["approved", "rejected"] | None
    escalation_id: str | None

    # --- bookkeeping ---
    current_step: str
    degraded: bool
    errors: Annotated[list[str], merge_list]
    agent_log: Annotated[list[dict], merge_list]
    final_message: str


def initial_state(**kwargs: Any) -> AgentCareState:
    base: AgentCareState = {
        "safety_verdict": "proceed",
        "blocked": False,
        "routing_confidence": 0.0,
        "routing_alternatives": [],
        "timeframe": None,
        "timeframe_note": "",
        "available_slots": [],
        "documents": [],
        "duplicates": [],
        "missing_documents": [],
        "reminders": [],
        "requires_approval": False,
        "current_step": "created",
        "degraded": False,
        "errors": [],
        "agent_log": [],
        "final_message": "",
    }
    base.update(kwargs)  # type: ignore[typeddict-item]
    return base
