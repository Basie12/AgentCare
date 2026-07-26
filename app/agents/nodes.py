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
from app.agents.state import AgentCareState
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

    # Low confidence is not a guess — it is a handoff to a human.
    if department is None or confidence < ROUTING_CONFIDENCE_THRESHOLD:
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
    writes=["available_slots", "appointment", "timeframe", "timeframe_note", "errors"],
)
def appointment_agent(state: AgentCareState) -> dict:
    run_id = state.get("workflow_run_id")
    department = state.get("department")
    patient_id = state.get("patient_id")
    request_text = state.get("request_text", "")

    if not department:
        return {"errors": ["appointment: no department resolved"], "current_step": "appointment_skipped"}

    # What did the patient actually ask for? Parsed deterministically — a
    # hallucinated date is worse than no date.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    timeframe = parse_timeframe(request_text, now)

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

    # Requested window came back empty — widen the search rather than fail.
    widened = False
    if not slots and timeframe.has_window:
        widened = True
        slots_result = hosp_tools.find_available_slots(
            _agent="appointment",
            _workflow_run_id=run_id,
            department_name=department,
            days_ahead=30,
            limit=12,
        )
        slots = slots_result.data if slots_result.ok else []

    if not slots:
        return {
            "available_slots": [],
            "timeframe": timeframe.to_dict(),
            "timeframe_note": f"no availability in {department}",
            "errors": [f"appointment: no open slots in {department}"],
            "current_step": "no_availability",
            "requires_approval": True,
            "approval_reason": f"No availability in {department}; staff to offer alternatives.",
        }

    ranked, note = select_slot(slots, timeframe)
    if widened:
        note = f"no availability within {timeframe.label}; searched further ahead instead"

    # Try the requested slot first, then walk the ranked candidates. A slot can
    # fail because it was taken concurrently, or because it clashes with an
    # appointment the patient already has.
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
        # Every candidate clashed — this needs a human, so record it as one.
        esc = ovs_tools.create_escalation(
            _agent="coordinator",
            _workflow_run_id=run_id,
            workflow_run_id=run_id,
            patient_id=patient_id,
            reason=(
                f"Automatic booking failed for {department} after "
                f"{len(attempts)} attempt(s). Last error: "
                f"{booking.error if booking else 'no candidates'}"
            ),
            category="booking_failed",
            severity="medium",
        )
        return {
            "available_slots": ranked[:8],
            "timeframe": timeframe.to_dict(),
            "timeframe_note": note,
            "errors": [f"appointment: exhausted {len(attempts)} slot(s)"] + attempts[:3],
            "current_step": "booking_failed",
            "requires_approval": True,
            "approval_reason": "Automatic booking failed; staff to book manually.",
            "escalation_id": esc.data["escalation_id"] if esc.ok else None,
        }

    # Did the slot we actually got honour the request?
    booked_at = datetime.fromisoformat(booking.data["start_time"])
    honoured = timeframe.contains(booked_at) if timeframe.specified else True
    if timeframe.specified and not honoured:
        note = f"{note} (booked outside the requested {timeframe.label})"

    return {
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

    if not appointment:
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


@agent_node(
    name="coordinator",
    reads=["appointment", "documents", "missing_documents", "reminders",
           "department", "timeframe"],
    writes=["final_message", "current_step"],
)
def coordinator_agent(state: AgentCareState) -> dict:
    facts = {
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
        "current_step": "completed",
        "_trace": trace,
    }