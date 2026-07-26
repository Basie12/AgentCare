"""Persistent entities for AgentCare.

Mirrors the suggested data model in the challenge brief (section 9) and adds
three tables that make agent behaviour auditable and reviewable:
AgentTrace, ToolInvocation and SafetyEvent.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def enum_col(enum_cls):
    """Persist enum *values* (lowercase) rather than member names."""
    return Enum(
        enum_cls,
        native_enum=False,
        values_callable=lambda e: [m.value for m in e],
        length=32,
    )


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class Role(str, enum.Enum):
    PATIENT = "patient"
    STAFF = "staff"
    ADMIN = "admin"


class SlotStatus(str, enum.Enum):
    OPEN = "open"
    HELD = "held"
    BOOKED = "booked"
    BLOCKED = "blocked"


class AppointmentStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class WorkflowStatus(str, enum.Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class EscalationStatus(str, enum.Enum):
    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReminderStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    SENT = "sent"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(
        enum_col(Role), nullable=False, default=Role.PATIENT)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    patient_profile: Mapped["PatientProfile"] = relationship(
        back_populates="user", uselist=False
    )


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    date_of_birth: Mapped[str | None] = mapped_column(String(32))
    phone: Mapped[str | None] = mapped_column(String(40))
    preferred_language: Mapped[str] = mapped_column(String(40), default="en")
    emergency_contact: Mapped[str | None] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="patient_profile")


# --------------------------------------------------------------------------
# Hospital directory
# --------------------------------------------------------------------------
class Department(Base):
    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # Documents a patient is expected to bring for this department.
    required_documents: Mapped[list] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    doctors: Mapped[list["Doctor"]] = relationship(back_populates="department")


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    department_id: Mapped[str] = mapped_column(ForeignKey("departments.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    department: Mapped[Department] = relationship(back_populates="doctors")
    slots: Mapped[list["AppointmentSlot"]] = relationship(back_populates="doctor")


class AppointmentSlot(Base):
    __tablename__ = "appointment_slots"
    __table_args__ = (
        UniqueConstraint("doctor_id", "start_time", name="uq_doctor_start"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.id"), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[SlotStatus] = mapped_column(
        enum_col(SlotStatus), default=SlotStatus.OPEN)

    doctor: Mapped[Doctor] = relationship(back_populates="slots")


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        # Hard guarantee against double-booking: only one live appointment
        # may reference a given slot. Enforced by the database, not by code.
        Index(
            "uq_slot_live_appointment",
            "slot_id",
            unique=True,
            sqlite_where=text("status IN ('pending','confirmed','rescheduled')"),
            postgresql_where=text("status IN ('pending','confirmed','rescheduled')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patient_profiles.id"), nullable=False)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.id"), nullable=False)
    slot_id: Mapped[str] = mapped_column(ForeignKey("appointment_slots.id"), nullable=False)
    status: Mapped[AppointmentStatus] = mapped_column(
        enum_col(AppointmentStatus), default=AppointmentStatus.PENDING
    )
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    slot: Mapped[AppointmentSlot] = relationship()
    doctor: Mapped[Doctor] = relationship()


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
class PatientDocument(Base):
    __tablename__ = "patient_documents"
    __table_args__ = (
        # Same file uploaded twice for the same patient is a duplicate.
        UniqueConstraint("patient_id", "checksum", name="uq_patient_checksum"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patient_profiles.id"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str] = mapped_column(String(80), default="unclassified")
    classification_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    classified_by: Mapped[str] = mapped_column(String(32), default="heuristic")
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_date: Mapped[datetime | None] = mapped_column(DateTime)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------
# Workflow, oversight and audit
# --------------------------------------------------------------------------
class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    thread_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    patient_id: Mapped[str | None] = mapped_column(ForeignKey("patient_profiles.id"))
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    current_step: Mapped[str] = mapped_column(String(60), default="created")
    # Full serialised graph state — survives process restart.
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[WorkflowStatus] = mapped_column(
        enum_col(WorkflowStatus), default=WorkflowStatus.RUNNING
    )
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patient_profiles.id"), nullable=False)
    appointment_id: Mapped[str | None] = mapped_column(ForeignKey("appointments.id"))
    reminder_type: Mapped[str] = mapped_column(String(60), default="appointment")
    message: Mapped[str] = mapped_column(Text, default="")
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[ReminderStatus] = mapped_column(
        enum_col(ReminderStatus), default=ReminderStatus.SCHEDULED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    patient_id: Mapped[str | None] = mapped_column(ForeignKey("patient_profiles.id"))
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(60), default="uncertain")
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    status: Mapped[EscalationStatus] = mapped_column(
        enum_col(EscalationStatus), default=EscalationStatus.OPEN
    )
    reviewed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    actor_id: Mapped[str | None] = mapped_column(String(32))
    actor_type: Mapped[str] = mapped_column(String(20), default="agent")  # user | agent | system
    action: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(60))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    event_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class AgentTrace(Base):
    """One row per agent node execution. Powers the trace viewer."""

    __tablename__ = "agent_traces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), index=True)
    agent_name: Mapped[str] = mapped_column(String(60), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    input_summary: Mapped[str] = mapped_column(Text, default="")
    output_summary: Mapped[str] = mapped_column(Text, default="")
    llm_model: Mapped[str | None] = mapped_column(String(80))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    used_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ToolInvocation(Base):
    """One row per tool call, written by the @tool decorator."""

    __tablename__ = "tool_invocations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workflow_run_id: Mapped[str | None] = mapped_column(String(32), index=True)
    agent_name: Mapped[str | None] = mapped_column(String(60))
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False)
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SafetyEvent(Base):
    """Every time a deterministic safety layer fires."""

    __tablename__ = "safety_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workflow_run_id: Mapped[str | None] = mapped_column(String(32), index=True)
    layer: Mapped[str] = mapped_column(String(40))  # pre_triage | output_guard | pii_redaction
    rule: Mapped[str] = mapped_column(String(120))
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    matched_excerpt: Mapped[str] = mapped_column(Text, default="")
    action_taken: Mapped[str] = mapped_column(String(60), default="blocked")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
