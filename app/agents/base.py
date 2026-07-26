"""Shared plumbing for agent nodes.

Every agent node is wrapped so that it writes an AgentTrace row: which agent
ran, what it received, what it produced, how long it took, whether it fell back
to deterministic logic. This is what the trace viewer renders.
"""
from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from app.agents.state import AgentCareState
from app.db.base import session_scope
from app.db.models import AgentTrace
from app.tools.registry import tools_for

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@functools.lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


def _summarise(value: object, limit: int = 800) -> str:
    try:
        return json.dumps(value, default=str)[:limit]
    except (TypeError, ValueError):
        return str(value)[:limit]


def agent_node(name: str, reads: list[str], writes: list[str]):
    """Declare an agent node.

    `reads`/`writes` document the state slice this agent owns, and are recorded
    in the trace so the handoff between agents is visible.
    """

    def decorator(fn: Callable[[AgentCareState], dict]):
        @functools.wraps(fn)
        def wrapper(state: AgentCareState) -> dict:
            started = time.perf_counter()
            run_id = state.get("workflow_run_id")
            error: str | None = None
            update: dict = {}

            try:
                update = fn(state) or {}
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent %s failed", name)
                error = f"{type(exc).__name__}: {exc}"
                update = {
                    "errors": [f"{name}: {error}"],
                    "degraded": True,
                }

            latency_ms = int((time.perf_counter() - started) * 1000)
            meta = update.pop("_trace", {}) if isinstance(update, dict) else {}

            try:
                with session_scope() as db:
                    db.add(
                        AgentTrace(
                            workflow_run_id=run_id,
                            agent_name=name,
                            step_index=len(state.get("agent_log", [])),
                            input_summary=_summarise({k: state.get(k) for k in reads}),
                            output_summary=_summarise(
                                {k: update.get(k) for k in writes if k in update}
                            ),
                            llm_model=meta.get("model"),
                            prompt_tokens=meta.get("prompt_tokens", 0),
                            completion_tokens=meta.get("completion_tokens", 0),
                            latency_ms=latency_ms,
                            used_fallback=bool(meta.get("used_fallback")),
                            error=error,
                        )
                    )
            except Exception:
                logger.exception("failed to persist trace for %s", name)

            update.setdefault("agent_log", []).append(
                {
                    "agent": name,
                    "latency_ms": latency_ms,
                    "used_fallback": bool(meta.get("used_fallback")),
                    "tools": [t.name for t in tools_for(name)],
                    "error": error,
                }
            )
            return update

        wrapper.agent_name = name  # type: ignore[attr-defined]
        wrapper.reads = reads  # type: ignore[attr-defined]
        wrapper.writes = writes  # type: ignore[attr-defined]
        return wrapper

    return decorator
