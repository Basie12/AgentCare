"""Appointment intent detection.

The core journey requires book, reschedule *and* cancel. A patient expresses
those in prose — "move my cardiology appointment to next week", "I need to
cancel Friday" — so the appointment agent has to know which operation is being
asked for before it touches the database.

Deterministic, like timeframe parsing: cancelling the wrong appointment because
a model misread the verb is not an acceptable failure mode, and this runs
inside the booking path.
"""
from __future__ import annotations

import re
from enum import Enum


class AppointmentIntent(str, Enum):
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"


# Order matters. "reschedule" wins over "cancel" because "cancel my Tuesday
# slot and rebook Thursday" is a reschedule, not a cancellation.
RESCHEDULE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(reschedul\w+|re-?book|rebook)\b", re.I),
    re.compile(r"\b(move|change|shift|switch|push back|bring forward)\b[^.]{0,40}\b(appointment|booking|slot|visit)\b", re.I),
    re.compile(r"\b(appointment|booking|slot|visit)\b[^.]{0,40}\b(to a different|to another|earlier|later)\b", re.I),
    re.compile(
        r"\bcan(?:'?t|not| ?no longer)\b[^.]{0,25}\b(make|attend|come|be there)\b"
        r"|\bunable to (?:attend|make|come)\b",
        re.I,
    ),
    re.compile(r"\b(cancel\w*)\b[^.]{0,60}\b(and (?:re)?book|instead|another|different)\b", re.I),
]

CANCEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcancel\w*\b", re.I),
    re.compile(r"\b(call off|drop|withdraw)\b[^.]{0,30}\b(appointment|booking|visit)\b", re.I),
    re.compile(r"\bno longer (?:need|want|require)\b[^.]{0,30}\b(appointment|booking|visit)\b", re.I),
    re.compile(r"\b(delete|remove)\b[^.]{0,30}\b(my )?(appointment|booking)\b", re.I),
]


def detect_intent(text: str) -> AppointmentIntent:
    """Default is BOOK — the common case and the safe one.

    Getting this wrong toward BOOK creates a spare appointment a human can
    remove. Getting it wrong toward CANCEL destroys care a patient was relying
    on, so cancellation requires an explicit cancellation verb.
    """
    if not text:
        return AppointmentIntent.BOOK

    for pattern in RESCHEDULE_PATTERNS:
        if pattern.search(text):
            return AppointmentIntent.RESCHEDULE

    for pattern in CANCEL_PATTERNS:
        if pattern.search(text):
            return AppointmentIntent.CANCEL

    return AppointmentIntent.BOOK
