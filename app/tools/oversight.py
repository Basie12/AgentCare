"""Escalation, reminder and follow-up tools."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import (
    Appointment,
    AppointmentSlot,
    Escalation,
    EscalationStatus,
    Reminder,
    ReminderStatus,
    SafetyEvent,
)
from app.tools.registry import ToolResult, tool


@tool(
    name="create_escalation",
    description="Create a human-review record that blocks or flags a workflow for staff attention.",
    agents=["safety", "coordinator", "routing"],
)
def create_escalation(
    workflow_run_id: str,
    reason: str,
    category: str = "uncertain",
    severity: str = "medium",
    patient_id: str | None = None,
) -> ToolResult:
    with session_scope() as db:
        esc = Escalation(
            workflow_run_id=workflow_run_id,
            patient_id=patient_id,
            reason=reason[:2000],
            category=category,
            severity=severity,
            status=EscalationStatus.OPEN,
        )
        db.add(esc)
        db.flush()
        return ToolResult(
            ok=True,
            data={"escalation_id": esc.id, "status": esc.status.value, "severity": severity},
        )


@tool(
    name="record_safety_event",
    description="Persist a safety-layer trigger for audit and review.",
    agents=["safety"],
)
def record_safety_event(
    workflow_run_id: str | None,
    layer: str,
    rule: str,
    severity: str = "medium",
    excerpt: str = "",
    action_taken: str = "blocked",
) -> ToolResult:
    with session_scope() as db:
        event = SafetyEvent(
            workflow_run_id=workflow_run_id,
            layer=layer,
            rule=rule,
            severity=severity,
            matched_excerpt=excerpt[:500],
            action_taken=action_taken,
        )
        db.add(event)
        db.flush()
        return ToolResult(ok=True, data={"safety_event_id": event.id, "rule": rule})


@tool(
    name="create_reminder",
    description="Schedule a reminder or follow-up task for a patient.",
    agents=["followup"],
)
def create_reminder(
    patient_id: str,
    message: str,
    scheduled_at: str | None = None,
    appointment_id: str | None = None,
    reminder_type: str = "appointment",
) -> ToolResult:
    when = (
        datetime.fromisoformat(scheduled_at)
        if scheduled_at
        else datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    )
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)

    with session_scope() as db:
        reminder = Reminder(
            patient_id=patient_id,
            appointment_id=appointment_id,
            reminder_type=reminder_type,
            message=message[:1000],
            scheduled_at=when,
            status=ReminderStatus.SCHEDULED,
        )
        db.add(reminder)
        db.flush()
        return ToolResult(
            ok=True,
            data={
                "reminder_id": reminder.id,
                "scheduled_at": reminder.scheduled_at.isoformat(),
                "type": reminder_type,
            },
        )


@tool(
    name="schedule_appointment_reminders",
    description="Create the standard reminder set for a booked appointment (24h before, plus post-visit follow-up).",
    agents=["followup"],
)
def schedule_appointment_reminders(patient_id: str, appointment_id: str) -> ToolResult:
    with session_scope() as db:
        appt = db.get(Appointment, appointment_id)
        if not appt:
            return ToolResult(ok=False, error=f"unknown appointment {appointment_id}")
        slot = db.get(AppointmentSlot, appt.slot_id)
        if not slot:
            return ToolResult(ok=False, error="appointment has no slot")

        created = []
        plan = [
            ("appointment", slot.start_time - timedelta(hours=24),
             "Reminder: your appointment is tomorrow. Please bring any requested documents."),
            ("check_in", slot.start_time - timedelta(hours=2),
             "Your appointment is in 2 hours. Please arrive 15 minutes early to check in."),
            ("post_visit", slot.start_time + timedelta(days=3),
             "Follow-up: please confirm whether you need a further appointment."),
        ]
        for rtype, when, message in plan:
            reminder = Reminder(
                patient_id=patient_id,
                appointment_id=appointment_id,
                reminder_type=rtype,
                message=message,
                scheduled_at=when,
                status=ReminderStatus.SCHEDULED,
            )
            db.add(reminder)
            db.flush()
            created.append({"id": reminder.id, "type": rtype, "at": when.isoformat()})

        return ToolResult(ok=True, data={"reminders": created})


@tool(
    name="find_incomplete_workflows",
    description="Detect workflows that stalled or appointments with no reminders scheduled.",
    agents=["followup"],
)
def find_incomplete_workflows(older_than_hours: int = 24) -> ToolResult:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=older_than_hours)
    from app.db.models import WorkflowRun, WorkflowStatus

    with session_scope() as db:
        stalled = db.scalars(
            select(WorkflowRun).where(
                WorkflowRun.status.in_(
                    [WorkflowStatus.RUNNING, WorkflowStatus.AWAITING_APPROVAL]
                ),
                WorkflowRun.updated_at < cutoff,
            )
        ).all()
        return ToolResult(
            ok=True,
            data=[
                {"workflow_run_id": w.id, "step": w.current_step, "status": w.status.value}
                for w in stalled
            ],
        )
