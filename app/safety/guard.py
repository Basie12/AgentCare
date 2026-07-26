"""Layer 2 — output guard, runs AFTER every LLM generation.

Nothing produced by a model reaches a patient without passing through here.
If the model drifts into clinical territory the text is suppressed and replaced
with an administrative-only message, and a SafetyEvent row is written.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Dosage expressed numerically — the clearest signal of prescribing behaviour.
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|cc|g|iu|units?)\b(?!\s*(?:file|document|upload))",
    re.I,
)

FREQUENCY_PATTERN = re.compile(
    r"\b(?:once|twice|three times|\d+\s*times)\s+(?:a|per)\s+(?:day|week)\b|\b(?:bid|tid|qid|q\d+h)\b",
    re.I,
)

# Common drug stems. Deliberately conservative — false positives are cheap,
# a leaked prescription is a disqualification.
DRUG_PATTERN = re.compile(
    r"\b\w*(?:cillin|mycin|statin|prazole|olol|sartan|pril|azepam|codone|profen)\b"
    r"|\b(?:aspirin|paracetamol|acetaminophen|ibuprofen|metformin|insulin|warfarin|"
    r"prednisone|amoxicillin|morphine|tramadol|nitroglycerin)\b",
    re.I,
)

DIAGNOSIS_PATTERN = re.compile(
    r"\b(?:you (?:have|are having|likely have|probably have|may have|might have)|"
    r"this (?:is|indicates|suggests|means you)|"
    r"(?:appears|seems) to be)\s+"
    r"(?:a |an |the )?(?:\w+\s+){0,3}"
    r"(?:cancer|infection|diabetes|hypertension|arrhythmia|angina|pneumonia|fracture|"
    r"stroke|attack|failure|disease|disorder|syndrome|condition|deficiency|tumou?r)\b",
    re.I,
)

TREATMENT_PATTERN = re.compile(
    r"\b(?:you should (?:take|start|stop|increase|decrease|continue)|"
    r"i recommend (?:taking|starting|stopping)|"
    r"(?:stop|discontinue) (?:taking|your) (?:medication|treatment))\b",
    re.I,
)

RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("dosage_assertion", DOSAGE_PATTERN, "critical"),
    ("dosage_frequency", FREQUENCY_PATTERN, "critical"),
    ("drug_named", DRUG_PATTERN, "critical"),
    ("diagnosis_assertion", DIAGNOSIS_PATTERN, "critical"),
    ("treatment_instruction", TREATMENT_PATTERN, "high"),
]

SAFE_REPLACEMENT = (
    "I can only help with the administrative side of your care — booking, "
    "routing, documents and reminders. For anything clinical, please speak with "
    "your care team. I've flagged this for hospital staff to review."
)


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    rule: str | None = None
    severity: str = "low"
    excerpt: str = ""


def guard_output(text: str) -> GuardResult:
    """Allowlist-first: administrative confirmations pass untouched."""
    if not text:
        return GuardResult(text="")

    for rule, pattern, severity in RULES:
        match = pattern.search(text)
        if match:
            return GuardResult(
                text=SAFE_REPLACEMENT,
                blocked=True,
                rule=rule,
                severity=severity,
                excerpt=match.group(0)[:200],
            )

    return GuardResult(text=text)
