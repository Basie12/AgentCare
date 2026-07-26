"""Tool registry.

Every agent-callable capability is registered here. The decorator gives each
tool, for free:
  * a ToolInvocation row (arguments, result, latency, success/failure)
  * an AuditEvent row
  * uniform error capture so a failing tool degrades the workflow instead of
    crashing it

Agents are only permitted to call tools bound to their name, which is what
makes the agent roles genuinely distinct rather than cosmetically named.
"""
from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.db.base import session_scope
from app.db.models import AuditEvent, ToolInvocation

logger = logging.getLogger(__name__)

REGISTRY: dict[str, "ToolSpec"] = {}


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    agents: set[str] = field(default_factory=set)

    def json_schema(self) -> dict:
        """Shape used when advertising tools to the LLM."""
        return {"name": self.name, "description": self.description}


class ToolPermissionError(RuntimeError):
    """Raised when an agent calls a tool it is not bound to."""


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "data": self.data, "error": self.error}


def _summarise(value: Any, limit: int = 600) -> str:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


def tool(name: str, description: str, agents: list[str]):
    """Register a function as an agent-callable tool.

    `agents` is the allowlist of agent names permitted to invoke it.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., ToolResult]:
        @functools.wraps(fn)
        def wrapper(
            *args: Any,
            _agent: str = "system",
            _workflow_run_id: str | None = None,
            _actor_id: str | None = None,
            **kwargs: Any,
        ) -> ToolResult:
            spec = REGISTRY[name]
            if _agent != "system" and _agent not in spec.agents:
                raise ToolPermissionError(
                    f"Agent '{_agent}' is not permitted to call tool '{name}'"
                )

            started = time.perf_counter()
            error: str | None = None
            result: ToolResult

            try:
                data = fn(*args, **kwargs)
                result = data if isinstance(data, ToolResult) else ToolResult(ok=True, data=data)
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                logger.exception("tool %s failed", name)
                error = f"{type(exc).__name__}: {exc}"
                result = ToolResult(ok=False, error=error)

            latency_ms = int((time.perf_counter() - started) * 1000)

            try:
                with session_scope() as db:
                    db.add(
                        ToolInvocation(
                            workflow_run_id=_workflow_run_id,
                            agent_name=_agent,
                            tool_name=name,
                            arguments={
                                k: _summarise(v, 200) for k, v in kwargs.items()
                            },
                            result_summary=_summarise(result.data if result.ok else result.error),
                            success=result.ok,
                            error=error,
                            latency_ms=latency_ms,
                        )
                    )
                    db.add(
                        AuditEvent(
                            actor_id=_actor_id or _agent,
                            actor_type="agent",
                            action=f"tool.{name}",
                            entity_type="workflow_run",
                            entity_id=_workflow_run_id,
                            event_metadata={
                                "success": result.ok,
                                "latency_ms": latency_ms,
                                "error": error,
                            },
                        )
                    )
            except Exception:  # audit must never break the workflow
                logger.exception("failed to persist audit for tool %s", name)

            return result

        REGISTRY[name] = ToolSpec(
            name=name, description=description, fn=wrapper, agents=set(agents)
        )
        return wrapper

    return decorator


def tools_for(agent: str) -> list[ToolSpec]:
    return [spec for spec in REGISTRY.values() if agent in spec.agents]


def call(name: str, *, _agent: str, **kwargs: Any) -> ToolResult:
    if name not in REGISTRY:
        return ToolResult(ok=False, error=f"unknown tool: {name}")
    return REGISTRY[name].fn(_agent=_agent, **kwargs)
