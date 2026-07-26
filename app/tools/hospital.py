"""Patient, department, availability and appointment tools.

Each of these reads or writes the database. None returns a canned response.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.base import session_scope
from app.db.models import (
    Appointment,
    AppointmentStatus,
    AppointmentSlot,
    Department,
    Doctor,
    PatientProfile,
    SlotStatus,
    User,
)
from app.tools.registry import ToolResult, tool


@tool(
    name="get_patient_record",
    description="Fetch a patient profile by patient id, returning contact and profile fields.",
    agents=["coordinator", "appointment", "document", "followup"],
)
def get_patient_record(patient_id: str) -> ToolResult:
    with session_scope() as db:
        profile = db.get(PatientProfile, patient_id)
        if not profile:
            return ToolResult(ok=False, error=f"no patient profile {patient_id}")
        user = db.get(User, profile.user_id)
        return ToolResult(
            ok=True,
            data={
                "patient_id": profile.id,
                "name": user.name if user else None,
                "email": user.email if user else None,
                "phone": profile.phone,
                "date_of_birth": profile.date_of_birth,
                "preferred_language": profile.preferred_language,
                "emergency_contact": profile.emergency_contact,
            },
        )


@tool(
    name="list_departments",
    description="List active hospital departments with their descriptions and required documents.",
    agents=["coordinator", "routing", "document"],
)
def list_departments() -> ToolResult:
    with session_scope() as db:
        rows = db.scalars(
            select(Department).where(Department.active.is_(True)).order_by(Department.name)
        ).all()
        return ToolResult(
            ok=True,
            data=[
                {
                    "id": d.id,
                    "name": d.name,
                    "description": d.description,
                    "required_documents": d.required_documents or [],
                }
                for d in rows
            ],
        )


@tool(
    name="find_available_slots",
    description="Find open appointment slots in a department within a date window.",
    agents=["appointment"],
)
def find_available_slots(
    department_name: str,
    days_ahead: int = 14,
    limit: int = 8,
    after: str | None = None,
    before: str | None = None,
) -> ToolResult:
    """`after`/`before` are ISO datetimes narrowing the search to a requested
    window. Both optional — omitted means "earliest available"."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    horizon = now + timedelta(days=days_ahead)

    if after:
        try:
            now = max(now, datetime.fromisoformat(after))
        except ValueError:
            pass
    if before:
        try:
            horizon = min(horizon, datetime.fromisoformat(before))
        except ValueError:
            pass

    with session_scope() as db:
        dept = db.scalar(select(Department).where(Department.name == department_name))
        if not dept:
            return ToolResult(ok=False, error=f"unknown department: {department_name}")

        rows = db.scalars(
            select(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .where(
                Doctor.department_id == dept.id,
                Doctor.active.is_(True),
                AppointmentSlot.status == SlotStatus.OPEN,
                AppointmentSlot.start_time >= now,
                AppointmentSlot.start_time <= horizon,
            )
            .order_by(AppointmentSlot.start_time)
            .limit(limit)
        ).all()

        return ToolResult(
            ok=True,
            data=[
                {
                    "slot_id": s.id,
                    "doctor_id": s.doctor_id,
                    "doctor_name": db.get(Doctor, s.doctor_id).name,
                    "department": dept.name,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat(),
                    "start_display": s.start_time.strftime("%a %d %b, %-I:%M %p"),
                }
                for s in rows
            ],
        )


@tool(
    name="book_appointment",
    description="Book a specific slot for a patient. Fails safely if the slot was taken.",
    agents=["appointment"],
)
def book_appointment(patient_id: str, slot_id: str, reason: str = "") -> ToolResult:
    with session_scope() as db:
        slot = db.get(AppointmentSlot, slot_id, with_for_update=False)
        if not slot:
            return ToolResult(ok=False, error=f"unknown slot {slot_id}")
        if slot.status is not SlotStatus.OPEN:
            return ToolResult(ok=False, error=f"slot {slot_id} is no longer open")

        # Conflict check: same patient already booked at this time.
        clash = db.scalar(
            select(Appointment)
            .join(AppointmentSlot, AppointmentSlot.id == Appointment.slot_id)
            .where(
                Appointment.patient_id == patient_id,
                Appointment.status.in_(
                    [
                        AppointmentStatus.PENDING,
                        AppointmentStatus.CONFIRMED,
                        AppointmentStatus.RESCHEDULED,
                    ]
                ),
                AppointmentSlot.start_time < slot.end_time,
                AppointmentSlot.end_time > slot.start_time,
            )
        )
        if clash:
            return ToolResult(
                ok=False,
                error=f"patient already has appointment {clash.id} overlapping this time",
            )

        appointment = Appointment(
            patient_id=patient_id,
            doctor_id=slot.doctor_id,
            slot_id=slot.id,
            status=AppointmentStatus.CONFIRMED,
            reason=reason[:500],
        )
        slot.status = SlotStatus.BOOKED
        db.add(appointment)

        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            return ToolResult(ok=False, error="slot was booked concurrently; pick another")

        doctor = db.get(Doctor, slot.doctor_id)
        dept = db.get(Department, doctor.department_id)
        return ToolResult(
            ok=True,
            data={
                "appointment_id": appointment.id,
                "status": appointment.status.value,
                "doctor_name": doctor.name,
                "department": dept.name,
                "start_time": slot.start_time.isoformat(),
                "end_time": slot.end_time.isoformat(),
                "start_display": slot.start_time.strftime("%a %d %b %Y, %-I:%M %p"),
            },
        )


@tool(
    name="reschedule_appointment",
    description="Move an existing appointment to a different open slot.",
    agents=["appointment"],
)
def reschedule_appointment(appointment_id: str, new_slot_id: str) -> ToolResult:
    with session_scope() as db:
        appt = db.get(Appointment, appointment_id)
        if not appt:
            return ToolResult(ok=False, error=f"unknown appointment {appointment_id}")
        if appt.status is AppointmentStatus.CANCELLED:
            return ToolResult(ok=False, error="cannot reschedule a cancelled appointment")

        new_slot = db.get(AppointmentSlot, new_slot_id)
        if not new_slot or new_slot.status is not SlotStatus.OPEN:
            return ToolResult(ok=False, error=f"slot {new_slot_id} is not available")

        old_slot = db.get(AppointmentSlot, appt.slot_id)
        if old_slot:
            old_slot.status = SlotStatus.OPEN

        appt.slot_id = new_slot.id
        appt.doctor_id = new_slot.doctor_id
        appt.status = AppointmentStatus.RESCHEDULED
        new_slot.status = SlotStatus.BOOKED
        db.flush()

        return ToolResult(
            ok=True,
            data={
                "appointment_id": appt.id,
                "status": appt.status.value,
                "start_time": new_slot.start_time.isoformat(),
                "start_display": new_slot.start_time.strftime("%a %d %b %Y, %-I:%M %p"),
                "previous_slot_released": bool(old_slot),
            },
        )


@tool(
    name="cancel_appointment",
    description="Cancel an appointment and release its slot back to the pool.",
    agents=["appointment"],
)
def cancel_appointment(appointment_id: str) -> ToolResult:
    with session_scope() as db:
        appt = db.get(Appointment, appointment_id)
        if not appt:
            return ToolResult(ok=False, error=f"unknown appointment {appointment_id}")

        slot = db.get(AppointmentSlot, appt.slot_id)
        if slot:
            slot.status = SlotStatus.OPEN
        appt.status = AppointmentStatus.CANCELLED
        db.flush()

        return ToolResult(
            ok=True,
            data={"appointment_id": appt.id, "status": appt.status.value},
        )
