"""Tests for the parts of AgentCare that must not silently break."""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agents.state import initial_state
from app.db.base import SessionLocal
from app.db.models import (
    Appointment,
    AppointmentStatus,
    Escalation,
    EscalationStatus,
    PatientProfile,
    Role,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.main import app
from app.safety.guard import guard_output
from app.safety.pii import redact
from app.safety.triage import TriageVerdict, triage
from app.services import resolve_escalation, start_workflow
from app.tools import hospital as hosp
from app.tools.registry import ToolPermissionError


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def patient(db):
    return db.scalars(select(PatientProfile)).first()


@pytest.fixture
def staff(db):
    return db.scalar(select(User).where(User.role == Role.STAFF))


# ---------------------------------------------------------------------------
# Safety — RULE-5 compliance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("I have chest pain and can't breathe", TriageVerdict.EMERGENCY),
        ("I want to kill myself", TriageVerdict.EMERGENCY),
        ("What dose of ibuprofen should I take?", TriageVerdict.CLINICAL_ADVICE),
        ("Do I have diabetes?", TriageVerdict.CLINICAL_ADVICE),
        ("Book my son a paediatrics appointment", TriageVerdict.SENSITIVE),
        ("Please reschedule my Cardiology appointment to Friday", TriageVerdict.PROCEED),
    ],
)
def test_triage_classifies_correctly(text, expected):
    assert triage(text).verdict is expected


@pytest.mark.parametrize(
    "text",
    [
        "Take 500 mg twice a day.",
        "You should start amoxicillin.",
        "You likely have a heart condition.",
        "I recommend taking metformin.",
    ],
)
def test_output_guard_blocks_clinical_text(text):
    result = guard_output(text)
    assert result.blocked
    assert "mg" not in result.text


def test_output_guard_allows_administrative_text():
    text = "Your Cardiology appointment with Dr. Rao is confirmed for Tuesday at 10:30."
    assert not guard_output(text).blocked


def test_pii_is_redacted_before_llm():
    result = redact("Reach me at 410-555-0198 or a@b.com, MRN: AB123456")
    assert "410-555-0198" not in result.text
    assert "a@b.com" not in result.text
    assert set(result.hits) >= {"PHONE", "EMAIL", "MRN"}
    assert result.restore(result.text).count("410-555-0198") == 1


# ---------------------------------------------------------------------------
# Tool permissions — agents are genuinely scoped
# ---------------------------------------------------------------------------
def test_agent_cannot_call_unbound_tool():
    with pytest.raises(ToolPermissionError):
        hosp.book_appointment(_agent="routing", patient_id="x", slot_id="y")


def test_tool_failure_is_captured_not_raised():
    result = hosp.book_appointment(_agent="appointment", patient_id="none", slot_id="none")
    assert result.ok is False
    assert result.error


# ---------------------------------------------------------------------------
# Workflow behaviour
# ---------------------------------------------------------------------------
def test_emergency_request_blocks_and_escalates(db, patient):
    run = start_workflow(
        db,
        patient_id=patient.id,
        actor_id=patient.user_id,
        request_text="I have severe chest pain right now",
    )
    assert run.status is WorkflowStatus.BLOCKED
    assert run.state["safety_verdict"] == "emergency"
    assert not run.state.get("appointment"), "a blocked workflow must not book anything"
    escalations = db.scalars(
        select(Escalation).where(Escalation.workflow_run_id == run.id)
    ).all()
    assert escalations, "an emergency must create an escalation record"


def test_happy_path_books_and_persists(db, patient):
    run = start_workflow(
        db,
        patient_id=patient.id,
        actor_id=patient.user_id,
        request_text="I need a Radiology appointment for a scheduled scan.",
    )
    assert run.status is WorkflowStatus.COMPLETED
    assert run.state["department"] == "Radiology"

    appt_id = run.state["appointment"]["appointment_id"]
    stored = db.get(Appointment, appt_id)
    assert stored is not None, "appointment must exist in the database, not just in state"
    assert stored.status is AppointmentStatus.CONFIRMED


def test_duplicate_document_is_detected(db, patient):
    payload = base64.b64encode(b"UNIQUE ECG TRACE FOR DUPLICATE TEST").decode()
    files = [{"filename": "trace.txt", "content_b64": payload}]

    first = start_workflow(
        db, patient_id=patient.id, actor_id=patient.user_id,
        request_text="ENT appointment please", uploaded_files=files,
    )
    second = start_workflow(
        db, patient_id=patient.id, actor_id=patient.user_id,
        request_text="Ophthalmology appointment please", uploaded_files=files,
    )
    assert len(first.state["documents"]) == 1
    assert len(second.state["duplicates"]) == 1
    assert second.state["documents"] == []


def test_interrupt_and_resume_from_checkpoint(db, patient, staff):
    run = start_workflow(
        db,
        patient_id=patient.id,
        actor_id=patient.user_id,
        request_text="I would like to arrange something about a matter please.",
    )
    assert run.status is WorkflowStatus.AWAITING_APPROVAL
    assert run.state["interrupt"]["type"] == "staff_approval_required"

    escalation = db.scalars(
        select(Escalation).where(
            Escalation.workflow_run_id == run.id,
            Escalation.status == EscalationStatus.OPEN,
        )
    ).first()
    assert escalation is not None

    resumed = resolve_escalation(
        db,
        escalation_id=escalation.id,
        reviewer_id=staff.id,
        decision="approved",
        note="Routing manually.",
        department="ENT",
    )
    assert resumed.status is WorkflowStatus.COMPLETED
    assert resumed.state["department"] == "ENT"
    assert resumed.state["approval_decision"] == "approved"

    db.refresh(escalation)
    assert escalation.status is EscalationStatus.APPROVED
    assert escalation.reviewed_by == staff.id


def _throwaway_patient(db, label: str) -> PatientProfile:
    """Isolated patient so this test does not inherit bookings from other tests."""
    from app.auth.security import hash_password

    user = User(
        name=f"Test {label}",
        email=f"{label}@test.local",
        password_hash=hash_password("x"),
        role=Role.PATIENT,
    )
    db.add(user)
    db.flush()
    profile = PatientProfile(user_id=user.id, preferred_language="en")
    db.add(profile)
    db.commit()
    return profile


def test_no_double_booking_of_a_slot(db):
    import uuid

    slots = hosp.find_available_slots(
        _agent="appointment", department_name="Dermatology", days_ahead=14, limit=5
    )
    slot_id = slots.data[0]["slot_id"]

    alice = _throwaway_patient(db, f"alice{uuid.uuid4().hex[:6]}")
    bob = _throwaway_patient(db, f"bob{uuid.uuid4().hex[:6]}")

    first = hosp.book_appointment(_agent="appointment", patient_id=alice.id, slot_id=slot_id)
    second = hosp.book_appointment(_agent="appointment", patient_id=bob.id, slot_id=slot_id)

    assert first.ok, f"first booking should succeed: {first.error}"
    assert not second.ok, "the same slot must not be bookable twice"


def test_patient_cannot_double_book_overlapping_times(db):
    import uuid

    profile = _throwaway_patient(db, f"clash{uuid.uuid4().hex[:6]}")
    slots = hosp.find_available_slots(
        _agent="appointment", department_name="Paediatrics", days_ahead=14, limit=5
    )
    first = hosp.book_appointment(
        _agent="appointment", patient_id=profile.id, slot_id=slots.data[0]["slot_id"]
    )
    assert first.ok

    # A different doctor at the same time must still be refused.
    same_time = [
        s for s in slots.data[1:]
        if s["start_time"] == slots.data[0]["start_time"]
    ]
    if same_time:
        clash = hosp.book_appointment(
            _agent="appointment", patient_id=profile.id, slot_id=same_time[0]["slot_id"]
        )
        assert not clash.ok, "a patient must not hold two appointments at the same time"


# ---------------------------------------------------------------------------
# RBAC — enforced in the backend, not the template
# ---------------------------------------------------------------------------
@pytest.fixture
def patient_client():
    client = TestClient(app)
    client.post("/login", data={"email": "patient@demo.local", "password": "patient123"})
    return client


@pytest.fixture
def staff_client():
    client = TestClient(app)
    client.post("/login", data={"email": "staff@demo.local", "password": "staff123"})
    return client


def test_anonymous_user_cannot_reach_staff_console():
    assert TestClient(app).get("/staff", follow_redirects=False).status_code in (302, 401)


def test_patient_cannot_reach_staff_console(patient_client):
    assert patient_client.get("/staff", follow_redirects=False).status_code == 403


def test_staff_can_reach_staff_console(staff_client):
    assert staff_client.get("/staff").status_code == 200


def test_patient_cannot_read_another_patients_workflow(db, patient_client):
    other = db.scalars(select(PatientProfile)).all()[1]
    run = start_workflow(
        db, patient_id=other.id, actor_id=other.user_id,
        request_text="Orthopaedics appointment for a routine review",
    )
    response = patient_client.get(f"/workflow/{run.id}", follow_redirects=False)
    assert response.status_code == 403


def test_wrong_password_is_rejected():
    client = TestClient(app)
    assert client.post(
        "/login", data={"email": "staff@demo.local", "password": "nope"}
    ).status_code == 401


# ---------------------------------------------------------------------------
# Provider swapping — changing LLM_PROVIDER must not require code changes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "provider,key_field,expected_sdk",
    [
        ("groq", "groq_api_key", "openai"),
        ("openai", "openai_api_key", "openai"),
        ("anthropic", "anthropic_api_key", "anthropic"),
        ("gemini", "google_api_key", "openai"),
        ("openrouter", "openrouter_api_key", "openai"),
    ],
)
def test_each_provider_builds_a_client(provider, key_field, expected_sdk):
    from app.agents import llm
    from app.config import settings as live

    original = (live.llm_provider, getattr(live, key_field))
    try:
        live.llm_provider = provider
        setattr(live, key_field, "test-key-not-real")
        llm.reset_client()

        assert llm.active_provider().sdk == expected_sdk
        assert llm.provider_is_configured()
        assert llm._get_client() is not None
    finally:
        live.llm_provider, _ = original
        setattr(live, key_field, original[1])
        llm.reset_client()


def test_ollama_needs_no_api_key():
    from app.agents import llm
    from app.config import settings as live

    original = live.llm_provider
    try:
        live.llm_provider = "ollama"
        llm.reset_client()
        assert llm.provider_is_configured(), "local provider must not require a key"
    finally:
        live.llm_provider = original
        llm.reset_client()


def test_unknown_provider_falls_back_instead_of_crashing():
    from app.agents import llm
    from app.config import settings as live

    original = live.llm_provider
    try:
        live.llm_provider = "not-a-real-provider"
        llm.reset_client()
        assert llm.active_provider() is llm.PROVIDERS["groq"]
    finally:
        live.llm_provider = original
        llm.reset_client()


def test_missing_key_degrades_and_never_raises():
    from app.agents import llm
    from app.config import settings as live

    original = (live.llm_provider, live.groq_api_key)
    try:
        live.llm_provider = "groq"
        live.groq_api_key = None
        llm.reset_client()

        response = llm.complete("system", "Call me on 410-555-0198")
        assert response.used_fallback is True
        assert response.text == ""
        assert "GROQ_API_KEY" in (response.error or "")
        # PII must still be scrubbed even on the failure path.
        assert "PHONE" in response.redaction_hits
    finally:
        live.llm_provider, live.groq_api_key = original
        llm.reset_client()


# ---------------------------------------------------------------------------
# Timeframe parsing — a stated preference must be honoured
# ---------------------------------------------------------------------------
from datetime import datetime as _dt, timedelta as _td  # noqa: E402

from app.agents.timeframe import parse_timeframe, select_slot  # noqa: E402

_NOW = _dt(2026, 7, 25, 14, 30)   # a Saturday


@pytest.mark.parametrize(
    "text,expected_start,expected_end",
    [
        ("Cardiology follow-up next week", _dt(2026, 7, 27), _dt(2026, 8, 2)),
        ("appointment the week after next", _dt(2026, 8, 3), _dt(2026, 8, 9)),
        ("something tomorrow", _dt(2026, 7, 26), _dt(2026, 7, 26)),
        ("next Friday please", _dt(2026, 7, 31), _dt(2026, 7, 31)),
        ("in two weeks", _dt(2026, 8, 3), _dt(2026, 8, 9)),
        ("on August 3", _dt(2026, 8, 3), _dt(2026, 8, 3)),
        ("appointment on 2026-08-12", _dt(2026, 8, 12), _dt(2026, 8, 12)),
    ],
)
def test_timeframe_windows(text, expected_start, expected_end):
    tf = parse_timeframe(text, _NOW)
    assert tf.start.date() == expected_start.date()
    assert tf.end.date() == expected_end.date()


@pytest.mark.parametrize(
    "text,part",
    [
        ("tomorrow morning", "morning"),
        ("Friday afternoon", "afternoon"),
        ("something in the evening", "evening"),
    ],
)
def test_timeframe_part_of_day(text, part):
    assert parse_timeframe(text, _NOW).part_of_day == part


def test_no_timeframe_means_no_preference():
    tf = parse_timeframe("I need to see a dermatologist", _NOW)
    assert not tf.specified
    assert tf.start is None


def test_timeframe_never_returns_a_past_window():
    tf = parse_timeframe("this week", _NOW)
    assert tf.start >= _NOW


def test_select_slot_prefers_the_requested_window():
    tf = parse_timeframe("next week", _NOW)
    slots = [
        {"slot_id": "too-soon", "start_time": _dt(2026, 7, 26, 9, 0).isoformat()},
        {"slot_id": "in-window", "start_time": _dt(2026, 7, 28, 9, 0).isoformat()},
        {"slot_id": "too-late", "start_time": _dt(2026, 9, 1, 9, 0).isoformat()},
    ]
    ranked, note = select_slot(slots, tf)
    assert ranked[0]["slot_id"] == "in-window"
    assert "matched" in note


def test_select_slot_falls_back_rather_than_failing():
    tf = parse_timeframe("next week", _NOW)
    slots = [{"slot_id": "far-off", "start_time": _dt(2026, 9, 1, 9, 0).isoformat()}]
    ranked, note = select_slot(slots, tf)
    assert ranked, "must still offer something rather than book nothing"
    assert "no availability" in note


def test_part_of_day_ranks_above_wrong_time_same_day():
    tf = parse_timeframe("Friday afternoon", _NOW)
    slots = [
        {"slot_id": "fri-am", "start_time": _dt(2026, 7, 31, 9, 0).isoformat()},
        {"slot_id": "fri-pm", "start_time": _dt(2026, 7, 31, 14, 0).isoformat()},
    ]
    ranked, _ = select_slot(slots, tf)
    assert ranked[0]["slot_id"] == "fri-pm"


def test_workflow_books_inside_the_requested_week(db):
    profile = _throwaway_patient(db, f"tf{__import__('uuid').uuid4().hex[:6]}")
    run = start_workflow(
        db,
        patient_id=profile.id,
        actor_id=profile.user_id,
        request_text="I need an Orthopaedics appointment next week",
    )
    appointment = run.state.get("appointment")
    assert appointment, "a booking should have been made"
    assert appointment["timeframe_honoured"] is True

    booked = _dt.fromisoformat(appointment["start_time"])
    tf = parse_timeframe("next week", _dt.now(__import__("datetime").timezone.utc).replace(tzinfo=None))
    assert tf.start <= booked <= tf.end


# ---------------------------------------------------------------------------
# Stale session must not cause a redirect loop
# ---------------------------------------------------------------------------
def test_stale_cookie_does_not_loop(db):
    """Reproduces the reseed scenario: a validly-signed token whose user is gone."""
    from app.auth.security import create_access_token

    client = TestClient(app)
    dead_token = create_access_token("user-id-that-does-not-exist", "patient")
    client.cookies.set("access_token", dead_token)

    # Must render the login page, not bounce to /dashboard.
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200
    assert "Sign in" in response.text

    # And the dead cookie must be cleared so the next request is clean.
    assert 'access_token=""' in response.headers.get("set-cookie", "") or \
           "access_token=;" in response.headers.get("set-cookie", "")


def test_protected_route_with_stale_cookie_clears_it(db):
    from app.auth.security import create_access_token

    client = TestClient(app)
    client.cookies.set("access_token", create_access_token("ghost-user", "staff"))

    response = client.get("/staff", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/"
    assert "access_token=" in response.headers.get("set-cookie", "")


def test_garbage_cookie_is_rejected_cleanly():
    client = TestClient(app)
    client.cookies.set("access_token", "not-a-real-jwt")
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200


def test_full_login_flow_still_works_after_the_fix():
    client = TestClient(app)
    assert client.get("/", follow_redirects=False).status_code == 200

    client.post("/login", data={"email": "patient@demo.local", "password": "patient123"})
    # A valid session should now redirect "/" onward to the dashboard.
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/dashboard"


# ---------------------------------------------------------------------------
# Routing: distinguish "LLM failed" from "LLM correctly said it doesn't know"
# ---------------------------------------------------------------------------
def _fake_llm(monkeypatch, *, text: str, used_fallback: bool = False):
    from app.agents import llm as llm_module
    from app.agents import nodes as nodes_module

    def _stub(system, user, **kwargs):
        return llm_module.LLMResponse(
            text=text,
            model="test-model",
            provider="test",
            prompt_tokens=10,
            completion_tokens=5,
            used_fallback=used_fallback,
        )

    monkeypatch.setattr(nodes_module, "complete", _stub)


def _route(db, monkeypatch, request_text: str):
    profile = _throwaway_patient(db, f"rt{__import__('uuid').uuid4().hex[:6]}")
    run = start_workflow(
        db, patient_id=profile.id, actor_id=profile.user_id, request_text=request_text
    )
    routing = next(
        (entry for entry in run.state["agent_log"] if entry["agent"] == "routing"), None
    )
    return run, routing


def test_deliberate_null_is_not_marked_as_fallback(db, monkeypatch):
    """The model saying 'I cannot tell' is a correct answer, not a failure."""
    _fake_llm(
        monkeypatch,
        text='{"department": null, "confidence": 0.1, "rationale": "Too vague to route.", "alternatives": []}',
    )
    run, routing = _route(db, monkeypatch, "Next week appointment")

    assert routing is not None
    assert routing["used_fallback"] is False, "a deliberate null must not read as fallback"
    assert run.state["department"] is None
    assert run.status is WorkflowStatus.AWAITING_APPROVAL


def test_unavailable_llm_does_mark_fallback(db, monkeypatch):
    _fake_llm(monkeypatch, text="", used_fallback=True)
    run, routing = _route(db, monkeypatch, "I need a Cardiology follow-up")

    assert routing["used_fallback"] is True
    assert run.state["department"] == "Cardiology", "keyword matcher should rescue this"


def test_hallucinated_department_is_discarded(db, monkeypatch):
    _fake_llm(
        monkeypatch,
        text='{"department": "Wizardry", "confidence": 0.99, "rationale": "made up", "alternatives": []}',
    )
    run, routing = _route(db, monkeypatch, "I need to see someone")

    assert run.state["department"] != "Wizardry"
    assert routing["used_fallback"] is True, "a hallucination is a genuine failure"


def test_keyword_matcher_refuses_weak_matches(db):
    from app.agents.nodes import _fallback_route

    departments = [
        {"name": "Cardiology", "description": "Heart and vascular appointments"},
        {"name": "ENT", "description": "Ear nose and throat clinic"},
    ]
    name, confidence, _ = _fallback_route("Next week appointment", departments)
    assert name is None, "must not invent a department from fuzzy noise"
    assert confidence == 0.0


def test_keyword_matcher_still_catches_explicit_names(db):
    from app.agents.nodes import _fallback_route

    departments = [{"name": "Cardiology", "description": "Heart and vascular"}]
    name, confidence, _ = _fallback_route("book me into cardiology", departments)
    assert name == "Cardiology"
    assert confidence >= 0.8


# ---------------------------------------------------------------------------
# The document agent must not report "fallback" when it simply had no work
# ---------------------------------------------------------------------------
# One payload that every agent can parse: safety reads "verdict", routing reads
# "department", document reads "document_type".
_ALL_AGENTS_JSON = (
    '{"verdict": "proceed", "department": "Radiology", "confidence": 0.95, '
    '"rationale": "administrative request", "alternatives": [], '
    '"document_type": "referral_letter"}'
)


def _doc_trace(run):
    return next(
        (entry for entry in run.state["agent_log"] if entry["agent"] == "document"), None
    )


def test_all_duplicates_is_not_a_fallback(db, monkeypatch):
    """Skipping classification because everything was a duplicate is not a failure."""
    _fake_llm(monkeypatch, text=_ALL_AGENTS_JSON)
    payload = base64.b64encode(b"REFERRAL LETTER unique-for-dup-trace-test").decode()
    files = [{"filename": "referral.txt", "content_b64": payload}]
    profile = _throwaway_patient(db, f"dup{__import__('uuid').uuid4().hex[:6]}")

    first = start_workflow(
        db, patient_id=profile.id, actor_id=profile.user_id,
        request_text="Radiology appointment please", uploaded_files=files,
    )
    assert len(first.state["documents"]) == 1

    second = start_workflow(
        db, patient_id=profile.id, actor_id=profile.user_id,
        request_text="Ophthalmology appointment please", uploaded_files=files,
    )
    assert len(second.state["duplicates"]) == 1
    assert _doc_trace(second)["used_fallback"] is False, (
        "an all-duplicate upload skips the LLM; that is not a fallback"
    )


def test_no_uploads_is_not_a_fallback(db, monkeypatch):
    _fake_llm(monkeypatch, text='{"department": "ENT", "confidence": 0.9, "rationale": "x"}')
    profile = _throwaway_patient(db, f"nof{__import__('uuid').uuid4().hex[:6]}")
    run = start_workflow(
        db, patient_id=profile.id, actor_id=profile.user_id,
        request_text="ENT appointment please",
    )
    assert _doc_trace(run)["used_fallback"] is False


def test_unclassifiable_document_does_report_fallback(db, monkeypatch):
    """When the model is consulted and cannot answer, that IS a fallback."""
    _fake_llm(monkeypatch, text="", used_fallback=True)
    payload = base64.b64encode(b"opaque bytes with no useful hints at all").decode()
    profile = _throwaway_patient(db, f"unc{__import__('uuid').uuid4().hex[:6]}")

    run = start_workflow(
        db, patient_id=profile.id, actor_id=profile.user_id,
        request_text="Dermatology appointment please",
        uploaded_files=[{"filename": "scan001.bin", "content_b64": payload}],
    )
    assert _doc_trace(run)["used_fallback"] is True