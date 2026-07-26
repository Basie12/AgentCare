"""Layer 1 — deterministic pre-filter, runs BEFORE any LLM call.

This is intentionally not an LLM. RULE-5 compliance must not depend on a model
behaving well, so emergency detection and clinical-advice detection are plain
pattern matching that always runs and cannot be prompt-injected away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TriageVerdict(str, Enum):
    PROCEED = "proceed"
    EMERGENCY = "emergency"
    CLINICAL_ADVICE = "clinical_advice"
    SENSITIVE = "sensitive"


# Red-flag phrasing that must halt the workflow and surface emergency guidance.
EMERGENCY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("chest_pain", re.compile(r"\bchest (pain|pressure|tightness)\b", re.I)),
    ("breathing", re.compile(r"\b(can'?t breathe|cannot breathe|difficulty breathing|shortness of breath|gasping)\b", re.I)),
    ("stroke", re.compile(r"\b(face drooping|slurred speech|sudden numbness|one side of my body)\b", re.I)),
    ("bleeding", re.compile(r"\b(heavy bleeding|bleeding (heavily|badly)|won'?t stop bleeding|hemorrhag)\b", re.I)),
    ("consciousness", re.compile(r"\b(unconscious|passed out|unresponsive|fainted and)\b", re.I)),
    ("self_harm", re.compile(r"\b(suicid\w*|kill myself|end my life|self[- ]harm|hurt myself)\b", re.I)),
    ("overdose", re.compile(r"\b(overdose|took too many pills|poison(ed|ing))\b", re.I)),
    ("severe_allergy", re.compile(r"\b(anaphyla\w+|throat closing|swelling of (the )?(throat|tongue))\b", re.I)),
    ("explicit", re.compile(r"\b(emergency|911|ambulance|urgent care now|life[- ]threatening)\b", re.I)),
]

# Requests for clinical judgement — always refused, always escalated.
CLINICAL_ADVICE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("diagnosis_request", re.compile(r"\b(what('s| is) wrong with me|do i have|am i having|diagnose|what disease|is this cancer)\b", re.I)),
    ("prescription_request", re.compile(r"\b(prescribe|prescription for|what (medicine|medication|drug) should i|can i take)\b", re.I)),
    ("dosage_request", re.compile(r"\b(how (much|many).{0,20}(should i take|do i take)|what dose|dosage|mg should i)\b", re.I)),
    ("treatment_request", re.compile(r"\b(how do i treat|what treatment|should i stop taking|is it safe to take)\b", re.I)),
]

SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("minor", re.compile(r"\b(my (child|son|daughter|baby|toddler)|under 18|paediatric|pediatric patient)\b", re.I)),
    ("mental_health", re.compile(r"\b(psychiatric|mental health crisis|psychosis|severe depression)\b", re.I)),
    ("termination", re.compile(r"\b(abortion|termination of pregnancy)\b", re.I)),
]


@dataclass
class TriageResult:
    verdict: TriageVerdict
    rule: str | None = None
    excerpt: str = ""
    severity: str = "low"

    @property
    def blocks_workflow(self) -> bool:
        return self.verdict in {TriageVerdict.EMERGENCY, TriageVerdict.CLINICAL_ADVICE}

    @property
    def requires_escalation(self) -> bool:
        return self.verdict is not TriageVerdict.PROCEED


EMERGENCY_MESSAGE = (
    "This request describes symptoms that may need immediate medical attention. "
    "AgentCare cannot assess symptoms. Please call your local emergency number "
    "(911 in the US) or go to the nearest emergency department now. "
    "A member of hospital staff has been notified of this request."
)

CLINICAL_REFUSAL_MESSAGE = (
    "AgentCare handles hospital administration only — registration, routing, "
    "appointments, documents and reminders. It cannot diagnose conditions, "
    "recommend medication or advise on dosage. Your question has been forwarded "
    "to hospital staff, and I can still book you an appointment with the right "
    "department if you'd like."
)


def _scan(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[str, str] | None:
    for rule, pattern in patterns:
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            return rule, text[start:end].strip()
    return None


def triage(text: str) -> TriageResult:
    """Order matters: emergencies outrank everything else."""
    if not text:
        return TriageResult(TriageVerdict.PROCEED)

    hit = _scan(text, EMERGENCY_PATTERNS)
    if hit:
        return TriageResult(TriageVerdict.EMERGENCY, hit[0], hit[1], severity="critical")

    hit = _scan(text, CLINICAL_ADVICE_PATTERNS)
    if hit:
        return TriageResult(TriageVerdict.CLINICAL_ADVICE, hit[0], hit[1], severity="high")

    hit = _scan(text, SENSITIVE_PATTERNS)
    if hit:
        return TriageResult(TriageVerdict.SENSITIVE, hit[0], hit[1], severity="medium")

    return TriageResult(TriageVerdict.PROCEED)
