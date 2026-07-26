"""The five agent nodes.

Each has: its own prompt file, its own tool allowlist, its own state slice, and
its own deterministic fallback for when the LLM is unavailable.

  safety      -> app/prompts/safety.md      -> create_escalation, record_safety_event
  routing     -> app/prompts/routing.md     -> list_departments, create_escalation
  appointment -> (no prompt; pure logic)    -> find_available_slots, book_appointment, ...
  document    -> app/prompts/document.md    -> check_document_duplicate, store_document, ...
  followup    -> app/prompts/followup.md    -> create_reminder, schedule_appointment_reminders
  coordinator -> app/prompts/coordinator.md -> get_patient_record, list_patient_documents
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
import re
from difflib import SequenceMatcher

from app.agents.base import agent_node, load_prompt
from app.agents.llm import complete
from app.agents.intent import AppointmentIntent, detect_intent
from app.agents.state import AgentCareState
from app.db.base import session_scope
from app.db.models import Appointment
from app.agents.timeframe import parse_timeframe, select_slot
from app.safety.guard import guard_output
from app.safety.triage import (
    CLINICAL_REFUSAL_MESSAGE,
    EMERGENCY_MESSAGE,
    TriageVerdict,
    triage,
)
from app.tools import documents as doc_tools
from app.tools import hospital as hosp_tools
from app.tools import oversight as ovs_tools

logger = logging.getLogger(__name__)

ROUTING_CONFIDENCE_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# 0. Profile agent — resolve the patient record before anything else runs
# ---------------------------------------------------------------------------
@agent_node(
    name="profile",
    reads=["patient_id", "request_text"],
    writes=["patient", "errors"],
)
def profile_agent(state: AgentCareState) -> dict:
    """Load the patient record the rest of the workflow will act on.

    Authentication proves *who* is asking; this proves there is a patient
    record to act on. Every downstream agent assumes `patient_id` resolves to a
    real profile — this is where that assumption is checked rather than
    inherited.
    """
    run_id = state.get("workflow_run_id")
    patient_id = state.get("patient_id")

    if not patient_id:
        return {
            "errors": ["profile: no patient id on the request"],
            "current_step": "profile_missing",
        }

    record = hosp_tools.get_patient_record(
        _agent="profile", _workflow_run_id=run_id, patient_id=patient_id
    )

    if not record.ok:
        return {
            "errors": [f"profile: {record.error}"],
            "current_step": "profile_missing",
            "requires_approval": True,
            "approval_reason": "Patient record could not be resolved; staff to verify identity.",
        }

    return {
        "patient": record.data,
        "current_step": "profile_resolved",
    }


# ---------------------------------------------------------------------------
# 1. Safety agent
# ---------------------------------------------------------------------------
@agent_node(
    name="safety",
    reads=["request_text", "patient_id"],
    writes=["safety_verdict", "safety_rule", "safety_message", "blocked", "escalation_id"],
)
def safety_agent(state: AgentCareState) -> dict:
    request = state.get("request_text", "")
    run_id = state.get("workflow_run_id")

    # Layer 1 — deterministic, always runs, cannot be prompt-injected away.
    result = triage(request)
    verdict = result.verdict
    rule = result.rule
    severity = result.severity
    trace: dict = {"used_fallback": True}

    # Layer 2 — the LLM may only ESCALATE the verdict, never downgrade it.
    if verdict is TriageVerdict.PROCEED:
        response = complete(load_prompt("safety"), request, json_mode=True)
        trace = {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "used_fallback": response.used_fallback,
        }
        parsed = response.as_json({"verdict": "proceed"})
        llm_verdict = str(parsed.get("verdict", "proceed")).lower()
        if llm_verdict in {"emergency", "clinical_advice", "sensitive"}:
            verdict = TriageVerdict(llm_verdict)
            rule = f"llm_{llm_verdict}"
            severity = "high" if llm_verdict != "emergency" else "critical"

    update: dict = {
        "safety_verdict": verdict.value,
        "safety_rule": rule,
        "current_step": "safety_checked",
        "_trace": trace,
    }

    if verdict is TriageVerdict.PROCEED:
        update["blocked"] = False
        return update

    ovs_tools.record_safety_event(
        _agent="safety",
        _workflow_run_id=run_id,
        workflow_run_id=run_id,
        layer="pre_triage",
        rule=rule or "unknown",
        severity=severity,
        excerpt=result.excerpt,
        action_taken="blocked" if result.blocks_workflow else "flagged",
    )

    esc = ovs_tools.create_escalation(
        _agent="safety",
        _workflow_run_id=run_id,
        workflow_run_id=run_id,
        patient_id=state.get("patient_id"),
        reason=f"Safety triage fired: {rule}",
        category=verdict.value,
        severity=severity,
    )
    if esc.ok:
        update["escalation_id"] = esc.data["escalation_id"]

    if verdict is TriageVerdict.EMERGENCY:
        update.update(blocked=True, safety_message=EMERGENCY_MESSAGE, final_message=EMERGENCY_MESSAGE)
    elif verdict is TriageVerdict.CLINICAL_ADVICE:
        update.update(
            blocked=True,
            safety_message=CLINICAL_REFUSAL_MESSAGE,
            final_message=CLINICAL_REFUSAL_MESSAGE,
        )
    else:  # sensitive — continue, but require a human to approve
        update.update(
            blocked=False,
            requires_approval=True,
            approval_reason=f"Sensitive request category: {rule}",
        )

    return update


# ---------------------------------------------------------------------------
# 1b. Intent agent — what operation is being requested?
# ---------------------------------------------------------------------------
@agent_node(
    name="intent",
    reads=["request_text"],
    writes=["appointment_intent", "timeframe"],
)
def intent_agent(state: AgentCareState) -> dict:
    """Classify the administrative operation, once, before routing.

    Deterministic on purpose. Cancelling the wrong appointment because a model
    misread a verb is not a recoverable error, and every downstream node reads
    this instead of re-deriving it.
    """
    request = state.get("request_text", "")
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    intent = detect_intent(request)
    timeframe = parse_timeframe(request, now)

    return {
        "appointment_intent": intent.value,
        "timeframe": timeframe.to_dict(),
        "current_step": f"intent_{intent.value}",
    }


# ---------------------------------------------------------------------------
# 2. Routing agent
# ---------------------------------------------------------------------------
def _fallback_route(request: str, departments: list[dict]) -> tuple[str | None, float, str]:
    """Deterministic routing used when the LLM is unavailable."""
    text = request.lower()
    best: tuple[str | None, float] = (None, 0.0)

    for dept in departments:
        name = dept["name"]
        score = 0.0
        # Word-boundary match: "ENT" must not match inside "appointment".
        if re.search(rf"\b{re.escape(name.lower())}\b", text):
            score = 0.9
        else:
            keywords = (dept.get("description") or "").lower().split()
            overlap = sum(1 for kw in keywords if len(kw) > 4 and kw in text)
            score = min(0.55, 0.15 * overlap)
            score = max(score, SequenceMatcher(None, name.lower(), text).ratio() * 0.4)
        if score > best[1]:
            best = (name, score)

    # Below this, the "match" is fuzzy noise. Saying "I don't know" is more
    # useful than naming a department at 0.10 confidence.
    if best[0] is None or best[1] < 0.30:
        return None, 0.0, "No department keyword matched the request."
    return best[0], round(best[1], 2), "Keyword match against department directory (no LLM)."


@agent_node(
    name="routing",
    reads=["request_text", "safety_verdict"],
    writes=["department", "routing_confidence", "routing_rationale", "requires_approval"],
)
def routing_agent(state: AgentCareState) -> dict:
    request = state.get("request_text", "")
    run_id = state.get("workflow_run_id")

    dept_result = hosp_tools.list_departments(_agent="routing", _workflow_run_id=run_id)
    departments = dept_result.data if dept_result.ok else []
    names = [d["name"] for d in departments]

    catalogue = "\n".join(f"- {d['name']}: {d.get('description', '')}" for d in departments)
    response = complete(
        load_prompt("routing"),
        f"Available departments:\n{catalogue}\n\nPatient request:\n{request}",
        json_mode=True,
    )
    trace = {
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "used_fallback": response.used_fallback,
    }

    parsed = response.as_json()
    llm_answered = not response.used_fallback and bool(response.text)

    department = parsed.get("department")
    confidence = float(parsed.get("confidence") or 0.0)
    rationale = parsed.get("rationale") or ""
    alternatives = parsed.get("alternatives") or []

    # A department outside the directory is a hallucination — discard it.
    hallucinated = bool(department) and department not in names
    if hallucinated:
        logger.warning("routing agent proposed unknown department %r", department)
        department = None
        confidence = 0.0
        rationale = "Model proposed a department outside the directory; discarded."

    # The keyword matcher is a *fallback*, not a second opinion. Reach for it
    # only when the LLM genuinely could not answer — either it was unavailable
    # or it hallucinated. An LLM that deliberately returns null is doing its
    # job, and overwriting that with a fuzzy string match produces a worse
    # answer and a misleading trace.
    if not llm_answered or hallucinated:
        fallback_dept, fallback_conf, fallback_rationale = _fallback_route(
            request, departments
        )
        trace["used_fallback"] = True
        if fallback_dept:
            department, confidence, rationale = (
                fallback_dept,
                fallback_conf,
                fallback_rationale,
            )
        elif not rationale:
            rationale = fallback_rationale
    elif department is None:
        # The model was available and chose "I cannot tell". That is a correct
        # outcome that routes to a human, not a degraded one.
        trace["used_fallback"] = False
        rationale = rationale or "No department could be determined from the request."

    update: dict = {
        "department": department,
        "routing_confidence": confidence,
        "routing_rationale": rationale,
        "routing_alternatives": [a for a in alternatives if a in names],
        "current_step": "routed",
        "degraded": bool(trace.get("used_fallback")),
        "_trace": trace,
    }

    # Cancel and reschedule operate on an appointment that already exists, and
    # that appointment already knows its department. Demanding a confident
    # route for "cancel my appointment" would escalate every cancellation to a
    # human for no reason. Intent was decided by the intent node.
    intent = state.get("appointment_intent", AppointmentIntent.BOOK.value)
    department_optional = intent in (
        AppointmentIntent.CANCEL.value,
        AppointmentIntent.RESCHEDULE.value,
    )

    # Low confidence is not a guess — it is a handoff to a human.
    if not department_optional and (
        department is None or confidence < ROUTING_CONFIDENCE_THRESHOLD
    ):
        esc = ovs_tools.create_escalation(
            _agent="routing",
            _workflow_run_id=run_id,
            workflow_run_id=run_id,
            patient_id=state.get("patient_id"),
            reason=(
                f"Routing confidence {confidence:.2f} below threshold "
                f"{ROUTING_CONFIDENCE_THRESHOLD}. Best guess: {department or 'none'}."
            ),
            category="routing_uncertain",
            severity="medium",
        )
        update["requires_approval"] = True
        update["approval_reason"] = "Department could not be determined confidently."
        if esc.ok:
            update["escalation_id"] = esc.data["escalation_id"]

    return update


# ---------------------------------------------------------------------------
# 3. Appointment agent
# ---------------------------------------------------------------------------
@agent_node(
    name="appointment",
    reads=["department", "patient_id", "selected_slot_id", "request_text"],
    writes=["available_slots", "appointment", "timeframe", "timeframe_note",
            "appointment_intent", "errors"],
)
def appointment_agent(state: AgentCareState) -> dict:
    run_id = state.get("workflow_run_id")
    patient_id = state.get("patient_id")
    request_text = state.get("request_text", "")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    timeframe = parse_timeframe(request_text, now)
    intent = AppointmentIntent(state.get("appointment_intent", AppointmentIntent.BOOK.value))

    if intent is AppointmentIntent.CANCEL:
        return _cancel_path(state, run_id, patient_id, intent)
    if intent is AppointmentIntent.RESCHEDULE:
        return _reschedule_path(state, run_id, patient_id, timeframe, now, intent)
    return _book_path(state, run_id, patient_id, timeframe, now, intent)


def _existing_appointments(run_id: str | None, patient_id: str) -> list[dict]:
    result = hosp_tools.list_patient_appointments(
        _agent="appointment",
        _workflow_run_id=run_id,
        patient_id=patient_id,
        active_only=True,
    )
    return result.data if result.ok else []


def _needs_human(reason: str, run_id: str | None, patient_id: str | None,
                 category: str, extra: dict) -> dict:
    esc = ovs_tools.create_escalation(
        _agent="coordinator",
        _workflow_run_id=run_id,
        workflow_run_id=run_id,
        patient_id=patient_id,
        reason=reason,
        category=category,
        severity="medium",
    )
    return {
        **extra,
        "requires_approval": True,
        "approval_reason": reason,
        "escalation_id": esc.data["escalation_id"] if esc.ok else None,
    }


# --- cancel ---------------------------------------------------------------
def _cancel_path(state, run_id, patient_id, intent) -> dict:
    existing = _existing_appointments(run_id, patient_id)

    if not existing:
        return {
            "appointment_intent": intent.value,
            "errors": ["appointment: nothing to cancel"],
            "current_step": "nothing_to_cancel",
        }

    # More than one live appointment and no way to tell which — ask a human
    # rather than guess. Cancelling the wrong one destroys care.
    if len(existing) > 1 and not state.get("selected_appointment_id"):
        return {
            "appointment_intent": intent.value,
            "available_slots": [],
            "current_step": "cancel_ambiguous",
            **_needs_human(
                f"Patient has {len(existing)} live appointments; cannot tell which to cancel.",
                run_id, patient_id, "cancel_ambiguous",
                {"existing_appointments": existing},
            ),
        }

    target = state.get("selected_appointment_id") or existing[0]["appointment_id"]
    result = hosp_tools.cancel_appointment(
        _agent="appointment", _workflow_run_id=run_id, appointment_id=target
    )
    if not result.ok:
        return {
            "appointment_intent": intent.value,
            "errors": [f"appointment: {result.error}"],
            "current_step": "cancel_failed",
        }

    cancelled = next(a for a in existing if a["appointment_id"] == target)
    return {
        "appointment_intent": intent.value,
        "appointment": {**cancelled, "status": "cancelled", "cancelled": True},
        "current_step": "cancelled",
    }


# --- reschedule -----------------------------------------------------------
def _reschedule_path(state, run_id, patient_id, timeframe, now, intent) -> dict:
    existing = _existing_appointments(run_id, patient_id)

    if not existing:
        # Nothing to move — treat it as a fresh booking rather than failing.
        outcome = _book_path(state, run_id, patient_id, timeframe, now, intent)
        outcome["timeframe_note"] = (
            "no existing appointment to reschedule; booked a new one instead"
        )
        return outcome

    if len(existing) > 1 and not state.get("selected_appointment_id"):
        return {
            "appointment_intent": intent.value,
            "current_step": "reschedule_ambiguous",
            **_needs_human(
                f"Patient has {len(existing)} live appointments; cannot tell which to move.",
                run_id, patient_id, "reschedule_ambiguous",
                {"existing_appointments": existing},
            ),
        }

    target_id = state.get("selected_appointment_id") or existing[0]["appointment_id"]
    target = next(a for a in existing if a["appointment_id"] == target_id)
    department = state.get("department") or target["department"]

    slots_result = hosp_tools.find_available_slots(
        _agent="appointment",
        _workflow_run_id=run_id,
        department_name=department,
        days_ahead=timeframe.days_ahead(now, default=21),
        limit=40 if timeframe.specified else 10,
        after=timeframe.start.isoformat() if timeframe.start else None,
        before=timeframe.end.isoformat() if timeframe.end else None,
    )
    slots = [s for s in (slots_result.data or []) if s["slot_id"] != target["slot_id"]]

    if not slots and timeframe.has_window:
        slots_result = hosp_tools.find_available_slots(
            _agent="appointment", _workflow_run_id=run_id,
            department_name=department, days_ahead=30, limit=12,
        )
        slots = [s for s in (slots_result.data or []) if s["slot_id"] != target["slot_id"]]

    if not slots:
        return {
            "appointment_intent": intent.value,
            "timeframe": timeframe.to_dict(),
            "current_step": "reschedule_no_availability",
            **_needs_human(
                f"No alternative slot available in {department}; staff to offer options.",
                run_id, patient_id, "no_availability", {"available_slots": []},
            ),
        }

    ranked, note = select_slot(slots, timeframe)
    moved = None
    for candidate in ranked:
        moved = hosp_tools.reschedule_appointment(
            _agent="appointment",
            _workflow_run_id=run_id,
            appointment_id=target_id,
            new_slot_id=candidate["slot_id"],
        )
        if moved.ok:
            break

    if moved is None or not moved.ok:
        return {
            "appointment_intent": intent.value,
            "available_slots": ranked[:8],
            "errors": [f"appointment: {moved.error if moved else 'no candidates'}"],
            "current_step": "reschedule_failed",
            **_needs_human(
                "Automatic reschedule failed; staff to move it manually.",
                run_id, patient_id, "reschedule_failed", {},
            ),
        }

    booked_at = datetime.fromisoformat(moved.data["start_time"])
    honoured = timeframe.contains(booked_at) if timeframe.specified else True
    return {
        "appointment_intent": intent.value,
        "available_slots": ranked[:8],
        "appointment": {
            **moved.data,
            "department": department,
            "doctor_name": moved.data.get("doctor_name") or target["doctor_name"],
            "previous_start_display": target["start_display"],
            "timeframe_honoured": honoured,
            "rescheduled": True,
        },
        "timeframe": timeframe.to_dict(),
        "timeframe_note": note,
        "current_step": "rescheduled",
    }


# --- book -----------------------------------------------------------------
def _book_path(state, run_id, patient_id, timeframe, now, intent) -> dict:
    department = state.get("department")
    request_text = state.get("request_text", "")

    if not department:
        return {
            "appointment_intent": intent.value,
            "errors": ["appointment: no department resolved"],
            "current_step": "appointment_skipped",
        }

    slots_result = hosp_tools.find_available_slots(
        _agent="appointment",
        _workflow_run_id=run_id,
        department_name=department,
        days_ahead=timeframe.days_ahead(now, default=14),
        limit=40 if timeframe.specified else 8,
        after=timeframe.start.isoformat() if timeframe.start else None,
        before=timeframe.end.isoformat() if timeframe.end else None,
    )
    slots = slots_result.data if slots_result.ok else []

    widened = False
    if not slots and timeframe.has_window:
        widened = True
        slots_result = hosp_tools.find_available_slots(
            _agent="appointment", _workflow_run_id=run_id,
            department_name=department, days_ahead=30, limit=12,
        )
        slots = slots_result.data if slots_result.ok else []

    if not slots:
        return {
            "appointment_intent": intent.value,
            "available_slots": [],
            "timeframe": timeframe.to_dict(),
            "timeframe_note": f"no availability in {department}",
            "errors": [f"appointment: no open slots in {department}"],
            "current_step": "no_availability",
            **_needs_human(
                f"No availability in {department}; staff to offer alternatives.",
                run_id, patient_id, "no_availability", {},
            ),
        }

    ranked, note = select_slot(slots, timeframe)
    if widened:
        note = f"no availability within {timeframe.label}; searched further ahead instead"

    preferred = state.get("selected_slot_id")
    candidates = [preferred] if preferred else []
    candidates += [s["slot_id"] for s in ranked if s["slot_id"] != preferred]

    booking = None
    attempts: list[str] = []
    for slot_id in candidates:
        booking = hosp_tools.book_appointment(
            _agent="appointment",
            _workflow_run_id=run_id,
            patient_id=patient_id,
            slot_id=slot_id,
            reason=request_text[:400],
        )
        if booking.ok:
            break
        attempts.append(f"{slot_id[:8]}: {booking.error}")

    if booking is None or not booking.ok:
        return {
            "appointment_intent": intent.value,
            "available_slots": ranked[:8],
            "timeframe": timeframe.to_dict(),
            "timeframe_note": note,
            "errors": [f"appointment: exhausted {len(attempts)} slot(s)"] + attempts[:3],
            "current_step": "booking_failed",
            **_needs_human(
                f"Automatic booking failed for {department} after {len(attempts)} attempt(s).",
                run_id, patient_id, "booking_failed", {},
            ),
        }

    booked_at = datetime.fromisoformat(booking.data["start_time"])
    honoured = timeframe.contains(booked_at) if timeframe.specified else True
    if timeframe.specified and not honoured:
        note = f"{note} (booked outside the requested {timeframe.label})"

    return {
        "appointment_intent": intent.value,
        "available_slots": ranked[:8],
        "appointment": {**booking.data, "timeframe_honoured": honoured},
        "timeframe": timeframe.to_dict(),
        "timeframe_note": note,
        "current_step": "booked",
    }


# ---------------------------------------------------------------------------
# 4. Document agent
# ---------------------------------------------------------------------------
@agent_node(
    name="document",
    reads=["uploaded_files", "patient_id", "department"],
    writes=["documents", "duplicates", "missing_documents"],
)
def document_agent(state: AgentCareState) -> dict:
    run_id = state.get("workflow_run_id")
    patient_id = state.get("patient_id")
    files = state.get("uploaded_files") or []

    stored: list[dict] = []
    duplicates: list[dict] = []
    # Starts False: "no classification was needed" is not a fallback. This only
    # flips when the model was actually consulted and could not answer.
    trace: dict = {"used_fallback": False}

    for item in files:
        filename = item.get("filename", "upload.bin")
        try:
            content = base64.b64decode(item.get("content_b64", ""))
        except Exception:  # noqa: BLE001
            continue
        if not content:
            continue

        checksum = doc_tools.compute_checksum(content)

        dup = doc_tools.check_document_duplicate(
            _agent="document",
            _workflow_run_id=run_id,
            patient_id=patient_id,
            checksum=checksum,
        )
        if dup.ok and dup.data.get("duplicate"):
            duplicates.append({"filename": filename, **dup.data})
            continue

        # Classify: LLM first, deterministic filename heuristic as fallback.
        preview = content[:400].decode("utf-8", errors="ignore")
        response = complete(
            load_prompt("document"),
            f"Filename: {filename}\nText preview:\n{preview}",
            json_mode=True,
        )
        parsed = response.as_json()
        doc_type = parsed.get("document_type")
        confidence = float(parsed.get("confidence") or 0.0)
        classified_by = "llm"

        if doc_type not in doc_tools.DOCUMENT_TYPES or confidence < 0.5:
            doc_type, confidence = doc_tools.classify_by_filename(filename, preview)
            classified_by = "heuristic"

        trace = {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "used_fallback": classified_by == "heuristic",
        }

        saved = doc_tools.store_document(
            _agent="document",
            _workflow_run_id=run_id,
            patient_id=patient_id,
            filename=filename,
            content=content,
            document_type=doc_type,
            confidence=confidence,
            classified_by=classified_by,
        )
        if saved.ok:
            stored.append({"filename": filename, **saved.data})

    missing: list[str] = []
    if state.get("department"):
        gaps = doc_tools.check_missing_documents(
            _agent="document",
            _workflow_run_id=run_id,
            patient_id=patient_id,
            department_name=state["department"],
        )
        if gaps.ok:
            missing = gaps.data.get("missing", [])

    return {
        "documents": stored,
        "duplicates": duplicates,
        "missing_documents": missing,
        "current_step": "documents_processed",
        "_trace": trace,
    }


# ---------------------------------------------------------------------------
# 5. Follow-up agent
# ---------------------------------------------------------------------------
@agent_node(
    name="followup",
    reads=["appointment", "patient_id", "missing_documents"],
    writes=["reminders"],
)
def followup_agent(state: AgentCareState) -> dict:
    run_id = state.get("workflow_run_id")
    patient_id = state.get("patient_id")
    appointment = state.get("appointment")

    if not appointment or appointment.get("cancelled"):
        return {"reminders": [], "current_step": "followup_skipped"}

    result = ovs_tools.schedule_appointment_reminders(
        _agent="followup",
        _workflow_run_id=run_id,
        patient_id=patient_id,
        appointment_id=appointment["appointment_id"],
    )
    reminders = result.data.get("reminders", []) if result.ok else []

    for missing in state.get("missing_documents") or []:
        extra = ovs_tools.create_reminder(
            _agent="followup",
            _workflow_run_id=run_id,
            patient_id=patient_id,
            appointment_id=appointment["appointment_id"],
            reminder_type="document",
            message=f"Please upload your {missing.replace('_', ' ')} before the appointment.",
        )
        if extra.ok:
            reminders.append(extra.data)

    return {"reminders": reminders, "current_step": "followup_scheduled"}


# ---------------------------------------------------------------------------
# 6. Coordinator (final message, guarded)
# ---------------------------------------------------------------------------
def _deterministic_summary(state: AgentCareState) -> str:
    appt = state.get("appointment")
    intent = state.get("appointment_intent", "book")

    if appt and appt.get("cancelled"):
        return (
            f"Your {appt.get('department', '')} appointment on "
            f"{appt.get('start_display', 'the scheduled date')} has been cancelled. "
            "Nothing further is needed from you."
        ).strip()

    if appt and appt.get("rescheduled"):
        parts = [
            f"Your {appt['department']} appointment has been moved to "
            f"{appt.get('start_display')}."
        ]
        if appt.get("previous_start_display"):
            parts.append(f"It was previously {appt['previous_start_display']}.")
        if state.get("reminders"):
            parts.append(f"{len(state['reminders'])} reminder(s) have been updated.")
        return " ".join(parts)

    if not appt and intent == "cancel":
        return (
            "There is no active appointment on your record to cancel. "
            "If you think this is wrong, hospital staff can check for you."
        )

    if not appt:
        return (
            "Your request has been recorded and passed to hospital staff. "
            "You will receive an update once it has been reviewed."
        )
    when = appt.get("start_display") or appt["start_time"].replace("T", " at ")
    parts = [
        f"Your {appt['department']} appointment with {appt['doctor_name']} is "
        f"{appt['status']} for {when}."
    ]
    timeframe = state.get("timeframe") or {}
    if timeframe.get("specified") and not appt.get("timeframe_honoured", True):
        parts.append(
            f"You asked for {timeframe.get('label')}; that was not available, "
            f"so this is the closest we could offer."
        )
    if state.get("duplicates"):
        parts.append(
            f"{len(state['duplicates'])} uploaded file(s) were already on record, so they were not stored again."
        )
    if state.get("documents"):
        parts.append(f"{len(state['documents'])} new document(s) were filed to your record.")
    if state.get("missing_documents"):
        readable = ", ".join(m.replace("_", " ") for m in state["missing_documents"])
        parts.append(f"Still needed before your visit: {readable}.")
    if state.get("reminders"):
        parts.append(f"{len(state['reminders'])} reminder(s) have been scheduled.")
    return " ".join(parts)


def validate_completion(state: AgentCareState) -> dict:
    """Check the workflow actually did what was asked, against persisted facts.

    The coordinator writes the message a patient reads, so it must not narrate
    success that did not happen. Each check compares intent to outcome; a
    failure is recorded and escalated rather than papered over.
    """
    intent = state.get("appointment_intent", "book")
    appointment = state.get("appointment") or {}
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- the requested operation actually happened ---
    if intent == "cancel":
        check(
            "cancellation_applied",
            bool(appointment.get("cancelled")) or state.get("current_step") == "nothing_to_cancel",
            "appointment cancelled, or there was nothing to cancel",
        )
    elif intent == "reschedule":
        check(
            "reschedule_applied",
            bool(appointment.get("rescheduled")) or bool(appointment.get("appointment_id")),
            "appointment moved to a new slot",
        )
    else:
        check(
            "appointment_booked",
            bool(appointment.get("appointment_id")),
            "an appointment id exists in persisted state",
        )

    # --- the booking is real, not just present in state ---
    if appointment.get("appointment_id") and not appointment.get("cancelled"):
        with session_scope() as db:
            stored = db.get(Appointment, appointment["appointment_id"])
            check(
                "appointment_persisted",
                stored is not None,
                "appointment row is readable from the database",
            )

    # --- every uploaded file was either stored or identified as a duplicate ---
    uploaded = len(state.get("uploaded_files") or [])
    if uploaded:
        accounted = len(state.get("documents") or []) + len(state.get("duplicates") or [])
        check(
            "documents_accounted_for",
            accounted == uploaded,
            f"{accounted} of {uploaded} uploaded file(s) stored or flagged duplicate",
        )

    # --- an active appointment should have reminders ---
    if appointment.get("appointment_id") and not appointment.get("cancelled"):
        check(
            "reminders_scheduled",
            bool(state.get("reminders")),
            "at least one reminder exists for the appointment",
        )

    # --- nothing silently failed upstream ---
    check(
        "no_agent_errors",
        not state.get("errors"),
        "no agent reported an error",
    )

    complete = all(item["ok"] for item in checks)
    return {"checks": checks, "complete": complete}


@agent_node(
    name="coordinator",
    reads=["appointment", "documents", "missing_documents", "reminders",
           "department", "timeframe", "appointment_intent"],
    writes=["final_message", "completion", "current_step"],
)
def coordinator_agent(state: AgentCareState) -> dict:
    completion = validate_completion(state)

    # A workflow that did not do what was asked gets a human, not a cheerful
    # confirmation message.
    escalation_id = state.get("escalation_id")
    if not completion["complete"]:
        failed = [c["check"] for c in completion["checks"] if not c["ok"]]
        esc = ovs_tools.create_escalation(
            _agent="coordinator",
            _workflow_run_id=state.get("workflow_run_id"),
            workflow_run_id=state.get("workflow_run_id"),
            patient_id=state.get("patient_id"),
            reason=f"Workflow completed with failed checks: {', '.join(failed)}",
            category="incomplete_workflow",
            severity="medium",
        )
        if esc.ok:
            escalation_id = esc.data["escalation_id"]

    facts = {
        "operation": state.get("appointment_intent", "book"),
        "department": state.get("department"),
        "appointment": state.get("appointment"),
        "requested_timeframe": (state.get("timeframe") or {}).get("label"),
        "timeframe_honoured": (state.get("appointment") or {}).get("timeframe_honoured"),
        "scheduling_note": state.get("timeframe_note"),
        "new_documents": [d.get("document_type") for d in state.get("documents", [])],
        "duplicate_count": len(state.get("duplicates", [])),
        "missing_documents": state.get("missing_documents", []),
        "reminder_count": len(state.get("reminders", [])),
    }

    response = complete(
        load_prompt("coordinator"),
        f"Persisted facts (use only these):\n{facts}",
    )
    trace = {
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "used_fallback": response.used_fallback,
    }

    message = response.text or _deterministic_summary(state)

    # Final gate: nothing reaches the patient unguarded.
    guarded = guard_output(message)
    if guarded.blocked:
        ovs_tools.record_safety_event(
            _agent="safety",
            _workflow_run_id=state.get("workflow_run_id"),
            workflow_run_id=state.get("workflow_run_id"),
            layer="output_guard",
            rule=guarded.rule or "unknown",
            severity=guarded.severity,
            excerpt=guarded.excerpt,
            action_taken="suppressed_and_replaced",
        )
        message = _deterministic_summary(state)

    return {
        "final_message": message,
        "completion": completion,
        "escalation_id": escalation_id,
        "requires_approval": state.get("requires_approval") or not completion["complete"],
        "current_step": "completed" if completion["complete"] else "completed_with_gaps",
        "_trace": trace,
    }
