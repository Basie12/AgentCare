"""Graph orchestration.

The workflow is a conditional state machine, not a chain:

              START
                |
            [safety] ------ blocked ------> [blocked_exit] -> END
                | proceed
            [routing] ---- low confidence / sensitive ----+
                |                                         |
                |                              [human_approval]  <-- interrupt()
                |                                    |        |
                |                            approved |        | rejected
                |<-----------------------------------+        v
                |                                        [rejected_exit] -> END
          [appointment]
                |
           [document]
                |
           [followup]
                |
         [coordinator] -> END

State is checkpointed to SQLite after every node. When `human_approval` calls
interrupt(), the process can be killed entirely; the workflow resumes from the
exact checkpoint once a staff member submits a decision.
"""
from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents.nodes import (
    appointment_agent,
    coordinator_agent,
    document_agent,
    followup_agent,
    routing_agent,
    safety_agent,
)
from app.agents.state import AgentCareState
from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal + approval nodes
# ---------------------------------------------------------------------------
def blocked_exit(state: AgentCareState) -> dict:
    return {
        "current_step": "blocked",
        "final_message": state.get("safety_message")
        or "This request needs a member of hospital staff.",
    }


def human_approval(state: AgentCareState) -> dict:
    """Pause the graph and wait for a real staff decision.

    interrupt() serialises the current state to the checkpointer and stops
    execution. Nothing further runs until resume_workflow() is called with a
    decision, which is written by an authenticated staff user.
    """
    decision = interrupt(
        {
            "type": "staff_approval_required",
            "workflow_run_id": state.get("workflow_run_id"),
            "reason": state.get("approval_reason"),
            "proposed_department": state.get("department"),
            "routing_confidence": state.get("routing_confidence"),
            "alternatives": state.get("routing_alternatives", []),
            "request_text": state.get("request_text"),
        }
    )

    # Everything below runs only after a human responds.
    if isinstance(decision, dict):
        verdict = decision.get("decision", "rejected")
        override_dept = decision.get("department")
    else:
        verdict = str(decision)
        override_dept = None

    update: dict = {
        "approval_decision": verdict,
        "requires_approval": False,
        "current_step": f"approval_{verdict}",
    }
    if verdict == "approved" and override_dept:
        update["department"] = override_dept
        update["routing_confidence"] = 1.0
        update["routing_rationale"] = "Department set by hospital staff on review."
    return update


def rejected_exit(state: AgentCareState) -> dict:
    return {
        "current_step": "rejected",
        "final_message": (
            "A member of hospital staff has reviewed your request and will "
            "contact you directly to arrange the next step."
        ),
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def after_safety(state: AgentCareState) -> str:
    return "blocked_exit" if state.get("blocked") else "routing"


def after_routing(state: AgentCareState) -> str:
    return "human_approval" if state.get("requires_approval") else "appointment"


def after_approval(state: AgentCareState) -> str:
    return "appointment" if state.get("approval_decision") == "approved" else "rejected_exit"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_graph(checkpointer: Any | None = None):
    builder = StateGraph(AgentCareState)

    builder.add_node("safety", safety_agent)
    builder.add_node("routing", routing_agent)
    builder.add_node("human_approval", human_approval)
    builder.add_node("appointment", appointment_agent)
    builder.add_node("document", document_agent)
    builder.add_node("followup", followup_agent)
    builder.add_node("coordinator", coordinator_agent)
    builder.add_node("blocked_exit", blocked_exit)
    builder.add_node("rejected_exit", rejected_exit)

    builder.add_edge(START, "safety")
    builder.add_conditional_edges(
        "safety", after_safety, {"blocked_exit": "blocked_exit", "routing": "routing"}
    )
    builder.add_conditional_edges(
        "routing",
        after_routing,
        {"human_approval": "human_approval", "appointment": "appointment"},
    )
    builder.add_conditional_edges(
        "human_approval",
        after_approval,
        {"appointment": "appointment", "rejected_exit": "rejected_exit"},
    )
    builder.add_edge("appointment", "document")
    builder.add_edge("document", "followup")
    builder.add_edge("followup", "coordinator")
    builder.add_edge("coordinator", END)
    builder.add_edge("blocked_exit", END)
    builder.add_edge("rejected_exit", END)

    return builder.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    """Compiled graph with a persistent checkpointer."""
    path = Path(settings.checkpoint_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return build_graph(checkpointer)


def graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def is_interrupted(result: dict) -> bool:
    return bool(result.get("__interrupt__"))


def interrupt_payload(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def resume_command(decision: str, department: str | None = None) -> Command:
    return Command(resume={"decision": decision, "department": department})
