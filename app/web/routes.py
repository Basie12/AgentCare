"""HTTP layer. Every route is thin: authenticate, authorise, delegate, render."""
from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import (
    assert_owns_patient,
    current_patient_profile,
    get_current_user,
    require_staff,
)
from app.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.config import settings
from app.db.base import get_db
from app.db.models import (
    AgentTrace,
    Appointment,
    AppointmentSlot,
    AuditEvent,
    Department,
    Doctor,
    Escalation,
    EscalationStatus,
    PatientDocument,
    PatientProfile,
    Reminder,
    Role,
    SafetyEvent,
    ToolInvocation,
    User,
    WorkflowRun,
)
from app.services import resolve_escalation, start_workflow
from app.tools import hospital as hosp_tools

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
router = APIRouter()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    """Only redirect to the dashboard if the session is genuinely valid.

    Checking merely that a cookie *exists* causes a redirect loop when the
    token is stale — e.g. after the database is reseeded and the user id it
    references no longer exists. Validate, and clear the cookie if it is dead.
    """
    token = request.cookies.get("access_token")
    if token:
        payload = decode_access_token(token)
        if payload and db.get(User, payload.get("sub")):
            return RedirectResponse("/dashboard", status_code=302)

        response = templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Your session has expired. Please sign in again."},
        )
        response.delete_cookie("access_token")
        return response

    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(User).where(User.email == email.lower().strip()))
    if not user or not verify_password(password, user.password_hash):
        db.add(
            AuditEvent(
                actor_id=None,
                actor_type="user",
                action="auth.login_failed",
                entity_type="user",
                entity_id=email[:64],
                event_metadata={},
            )
        )
        db.commit()
        return templates.TemplateResponse(
            request, "login.html", {"error": "Email or password is incorrect."}, status_code=401
        )

    db.add(
        AuditEvent(
            actor_id=user.id,
            actor_type="user",
            action="auth.login",
            entity_type="user",
            entity_id=user.id,
            event_metadata={"role": user.role.value},
        )
    )
    db.commit()

    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(
        "access_token",
        create_access_token(user.id, user.role.value),
        httponly=True,
        samesite="lax",
        secure=settings.environment == "production",
    )
    return response


# ---------------------------------------------------------------------------
# Registration  (required scope step 1: Patient Registration)
# ---------------------------------------------------------------------------
@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse(request, "register.html", {"error": None, "form": {}})


@router.post("/register")
def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    date_of_birth: str = Form(default=""),
    phone: str = Form(default=""),
    preferred_language: str = Form(default="en"),
    emergency_contact: str = Form(default=""),
    db: Session = Depends(get_db),
):
    form = {
        "name": name, "email": email, "date_of_birth": date_of_birth,
        "phone": phone, "preferred_language": preferred_language,
        "emergency_contact": emergency_contact,
    }

    def reject(message: str):
        return templates.TemplateResponse(
            request, "register.html", {"error": message, "form": form}, status_code=400
        )

    email = email.lower().strip()
    if len(password) < 8:
        return reject("Password must be at least 8 characters.")
    if not name.strip():
        return reject("Please enter your name.")
    if db.scalar(select(User).where(User.email == email)):
        return reject("An account already exists for that email. Try signing in.")

    user = User(
        name=name.strip()[:160],
        email=email,
        password_hash=hash_password(password),
        role=Role.PATIENT,
    )
    db.add(user)
    db.flush()

    db.add(
        PatientProfile(
            user_id=user.id,
            date_of_birth=date_of_birth.strip() or None,
            phone=phone.strip() or None,
            preferred_language=preferred_language.strip() or "en",
            emergency_contact=emergency_contact.strip() or None,
        )
    )
    db.add(
        AuditEvent(
            actor_id=user.id,
            actor_type="user",
            action="patient.registered",
            entity_type="user",
            entity_id=user.id,
            event_metadata={"role": Role.PATIENT.value},
        )
    )
    db.commit()

    response = RedirectResponse("/patient", status_code=302)
    response.set_cookie(
        "access_token",
        create_access_token(user.id, user.role.value),
        httponly=True,
        samesite="lax",
        secure=settings.environment == "production",
    )
    return response


@router.post("/patient/profile")
def update_profile(
    date_of_birth: str = Form(default=""),
    phone: str = Form(default=""),
    preferred_language: str = Form(default="en"),
    emergency_contact: str = Form(default=""),
    user: User = Depends(get_current_user),
    profile: PatientProfile = Depends(current_patient_profile),
    db: Session = Depends(get_db),
):
    profile.date_of_birth = date_of_birth.strip() or None
    profile.phone = phone.strip() or None
    profile.preferred_language = preferred_language.strip() or "en"
    profile.emergency_contact = emergency_contact.strip() or None
    db.add(
        AuditEvent(
            actor_id=user.id, actor_type="user", action="patient.profile_updated",
            entity_type="patient_profile", entity_id=profile.id, event_metadata={},
        )
    )
    db.commit()
    return RedirectResponse("/patient", status_code=302)


@router.get("/logout")
def logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("access_token")
    return response


@router.get("/dashboard")
def dashboard(user: User = Depends(get_current_user)):
    target = "/staff" if user.role in (Role.STAFF, Role.ADMIN) else "/patient"
    return RedirectResponse(target, status_code=302)


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------
@router.get("/patient", response_class=HTMLResponse)
def patient_home(
    request: Request,
    user: User = Depends(get_current_user),
    profile: PatientProfile = Depends(current_patient_profile),
    db: Session = Depends(get_db),
):
    runs = db.scalars(
        select(WorkflowRun)
        .where(WorkflowRun.patient_id == profile.id)
        .order_by(WorkflowRun.created_at.desc())
        .limit(15)
    ).all()

    appointments = db.execute(
        select(Appointment, AppointmentSlot, Doctor)
        .join(AppointmentSlot, AppointmentSlot.id == Appointment.slot_id)
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .where(Appointment.patient_id == profile.id)
        .order_by(AppointmentSlot.start_time)
    ).all()

    documents = db.scalars(
        select(PatientDocument)
        .where(PatientDocument.patient_id == profile.id)
        .order_by(PatientDocument.created_at.desc())
    ).all()

    reminders = db.scalars(
        select(Reminder)
        .where(Reminder.patient_id == profile.id)
        .order_by(Reminder.scheduled_at)
        .limit(20)
    ).all()

    return templates.TemplateResponse(
        request,
        "patient.html",
        {
            "user": user,
            "profile": profile,
            "runs": runs,
            "appointments": appointments,
            "documents": documents,
            "reminders": reminders,
        },
    )


@router.post("/patient/request")
async def submit_request(
    request_text: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    user: User = Depends(get_current_user),
    profile: PatientProfile = Depends(current_patient_profile),
    db: Session = Depends(get_db),
):
    uploads = []
    for upload in files or []:
        if not upload.filename:
            continue
        content = await upload.read()
        if content:
            uploads.append(
                {
                    "filename": upload.filename,
                    "content_b64": base64.b64encode(content).decode(),
                }
            )

    run = start_workflow(
        db,
        patient_id=profile.id,
        actor_id=user.id,
        request_text=request_text,
        uploaded_files=uploads,
    )
    return RedirectResponse(f"/workflow/{run.id}", status_code=302)


@router.get("/workflow/{run_id}", response_class=HTMLResponse)
def workflow_detail(
    request: Request,
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = db.get(WorkflowRun, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workflow not found")
    assert_owns_patient(user, run.patient_id, db)

    traces = db.scalars(
        select(AgentTrace)
        .where(AgentTrace.workflow_run_id == run_id)
        .order_by(AgentTrace.created_at)
    ).all()
    tools = db.scalars(
        select(ToolInvocation)
        .where(ToolInvocation.workflow_run_id == run_id)
        .order_by(ToolInvocation.created_at)
    ).all()
    safety_events = db.scalars(
        select(SafetyEvent).where(SafetyEvent.workflow_run_id == run_id)
    ).all()

    return templates.TemplateResponse(
        request,
        "workflow.html",
        {
            "user": user,
            "run": run,
            "state": run.state or {},
            "traces": traces,
            "tools": tools,
            "safety_events": safety_events,
        },
    )


# ---------------------------------------------------------------------------
# Staff
# ---------------------------------------------------------------------------
@router.get("/staff", response_class=HTMLResponse)
def staff_home(
    request: Request,
    user: User = Depends(require_staff),
    db: Session = Depends(get_db),
):
    open_escalations = db.scalars(
        select(Escalation)
        .where(Escalation.status == EscalationStatus.OPEN)
        .order_by(Escalation.created_at.desc())
    ).all()
    resolved = db.scalars(
        select(Escalation)
        .where(Escalation.status != EscalationStatus.OPEN)
        .order_by(Escalation.created_at.desc())
        .limit(10)
    ).all()
    runs = db.scalars(
        select(WorkflowRun).order_by(WorkflowRun.created_at.desc()).limit(20)
    ).all()
    audit = db.scalars(
        select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(30)
    ).all()

    run_map = {r.id: r for r in db.scalars(select(WorkflowRun)).all()}

    return templates.TemplateResponse(
        request,
        "staff.html",
        {
            "user": user,
            "open_escalations": open_escalations,
            "resolved": resolved,
            "runs": runs,
            "audit": audit,
            "run_map": run_map,
        },
    )


@router.post("/staff/escalation/{escalation_id}")
def review_escalation(
    escalation_id: str,
    decision: str = Form(...),
    note: str = Form(default=""),
    department: str = Form(default=""),
    user: User = Depends(require_staff),
    db: Session = Depends(get_db),
):
    if decision not in {"approved", "rejected"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "decision must be approved or rejected")

    run = resolve_escalation(
        db,
        escalation_id=escalation_id,
        reviewer_id=user.id,
        decision=decision,
        note=note or None,
        department=department or None,
    )
    if run is None:
        return RedirectResponse("/staff", status_code=302)
    return RedirectResponse(f"/workflow/{run.id}", status_code=302)


# ---------------------------------------------------------------------------
# Reschedule and cancel  (core journey step 5)
# ---------------------------------------------------------------------------
def _load_owned_appointment(appointment_id: str, user: User, db: Session) -> Appointment:
    appointment = db.get(Appointment, appointment_id)
    if not appointment:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found")
    assert_owns_patient(user, appointment.patient_id, db)
    return appointment


@router.get("/appointment/{appointment_id}/reschedule", response_class=HTMLResponse)
def reschedule_form(
    request: Request,
    appointment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    appointment = _load_owned_appointment(appointment_id, user, db)
    doctor = db.get(Doctor, appointment.doctor_id)
    department = db.get(Department, doctor.department_id)
    slot = db.get(AppointmentSlot, appointment.slot_id)

    offered = hosp_tools.find_available_slots(
        _agent="appointment",
        _actor_id=user.id,
        department_name=department.name,
        days_ahead=30,
        limit=24,
    )
    slots = [s for s in (offered.data or []) if s["slot_id"] != appointment.slot_id]

    return templates.TemplateResponse(
        request,
        "reschedule.html",
        {
            "user": user,
            "appointment": appointment,
            "department": department,
            "doctor": doctor,
            "current_slot": slot,
            "slots": slots,
        },
    )


@router.post("/appointment/{appointment_id}/reschedule")
def reschedule(
    appointment_id: str,
    slot_id: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _load_owned_appointment(appointment_id, user, db)
    result = hosp_tools.reschedule_appointment(
        _agent="appointment",
        _actor_id=user.id,
        appointment_id=appointment_id,
        new_slot_id=slot_id,
    )
    if not result.ok:
        raise HTTPException(status.HTTP_409_CONFLICT, result.error or "Could not reschedule")
    return RedirectResponse("/patient", status_code=302)


@router.post("/appointment/{appointment_id}/cancel")
def cancel(
    appointment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _load_owned_appointment(appointment_id, user, db)
    result = hosp_tools.cancel_appointment(
        _agent="appointment", _actor_id=user.id, appointment_id=appointment_id
    )
    if not result.ok:
        raise HTTPException(status.HTTP_409_CONFLICT, result.error or "Could not cancel")
    return RedirectResponse("/patient", status_code=302)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": settings.llm_enabled,
        "environment": settings.environment,
    }
