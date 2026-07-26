"""PII redaction.

Every free-text payload is redacted *before* it is sent to an external LLM.
Redaction is reversible within a request via the returned token map, so agents
can still reference "the patient's phone number" without the value leaving
the process.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:MRN|mrn)[:\s#-]*([A-Z0-9]{6,12})\b")),
    ("DOB", re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")),
    ("INSURANCE_ID", re.compile(r"\b[A-Z]{3}\d{9,12}\b")),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b")),
]


@dataclass
class RedactionResult:
    text: str
    token_map: dict[str, str] = field(default_factory=dict)
    hits: list[str] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.hits)

    def restore(self, text: str) -> str:
        """Put real values back into an LLM response before showing the patient."""
        for token, original in self.token_map.items():
            text = text.replace(token, original)
        return text


def redact(text: str | None) -> RedactionResult:
    if not text:
        return RedactionResult(text="")

    token_map: dict[str, str] = {}
    hits: list[str] = []
    counters: dict[str, int] = {}
    working = text

    for label, pattern in PATTERNS:
        def _sub(match: re.Match[str]) -> str:
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            token_map[token] = match.group(0)
            hits.append(label)
            return token

        working = pattern.sub(_sub, working)

    return RedactionResult(text=working, token_map=token_map, hits=hits)
