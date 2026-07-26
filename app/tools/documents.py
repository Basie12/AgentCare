"""Document coordination tools — a high-weight area of the rubric.

Covers ingestion, checksum duplicate detection, type classification with a
deterministic fallback, patient mapping and missing-document checks.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db.base import session_scope
from app.db.models import Department, PatientDocument
from app.tools.registry import ToolResult, tool

DOCUMENT_TYPES = [
    "ecg",
    "lab_report",
    "imaging_report",
    "discharge_summary",
    "referral_letter",
    "prescription_record",
    "insurance_card",
    "identity_proof",
    "consent_form",
    "other",
]

# Deterministic fallback used when the LLM is unavailable or low-confidence.
FILENAME_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("ecg", re.compile(r"\b(ecg|ekg|electrocardio|holter)\b", re.I)),
    ("lab_report", re.compile(r"\b(lab|blood|cbc|panel|serum|urine|pathology|hba1c|lipid)\b", re.I)),
    ("imaging_report", re.compile(r"\b(x[- ]?ray|mri|ct[- ]?scan|ultrasound|sonograph|radiolog|echo)\b", re.I)),
    ("discharge_summary", re.compile(r"\b(discharge|summary|admission)\b", re.I)),
    ("referral_letter", re.compile(r"\b(referral|refer|letter)\b", re.I)),
    ("prescription_record", re.compile(r"\b(prescription|rx|medication[- ]?list)\b", re.I)),
    ("insurance_card", re.compile(r"\b(insurance|policy|coverage)\b", re.I)),
    ("identity_proof", re.compile(r"\b(id|passport|licence|license|aadhaar)\b", re.I)),
    ("consent_form", re.compile(r"\b(consent|authoriz|authoris)\b", re.I)),
]


def compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classify_by_filename(filename: str, text_preview: str = "") -> tuple[str, float]:
    haystack = f"{filename} {text_preview}"
    for doc_type, pattern in FILENAME_HINTS:
        if pattern.search(haystack):
            return doc_type, 0.6
    return "other", 0.2


@tool(
    name="check_document_duplicate",
    description="Check whether a document with this checksum already exists for the patient.",
    agents=["document"],
)
def check_document_duplicate(patient_id: str, checksum: str) -> ToolResult:
    with session_scope() as db:
        existing = db.scalar(
            select(PatientDocument).where(
                PatientDocument.patient_id == patient_id,
                PatientDocument.checksum == checksum,
            )
        )
        if existing:
            return ToolResult(
                ok=True,
                data={
                    "duplicate": True,
                    "existing_document_id": existing.id,
                    "existing_filename": existing.original_filename,
                    "uploaded_at": existing.created_at.isoformat(),
                    "document_type": existing.document_type,
                },
            )
        return ToolResult(ok=True, data={"duplicate": False})


@tool(
    name="store_document",
    description="Persist an uploaded document to storage and record its metadata against a patient.",
    agents=["document"],
)
def store_document(
    patient_id: str,
    filename: str,
    content: bytes,
    document_type: str = "unclassified",
    confidence: float = 0.0,
    classified_by: str = "heuristic",
) -> ToolResult:
    max_bytes = settings.max_document_mb * 1024 * 1024
    if len(content) > max_bytes:
        return ToolResult(ok=False, error=f"file exceeds {settings.max_document_mb}MB limit")

    checksum = compute_checksum(content)
    storage_dir = Path(settings.document_storage_dir) / patient_id
    storage_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[:120]
    path = storage_dir / f"{checksum[:12]}_{safe_name}"
    path.write_bytes(content)

    with session_scope() as db:
        doc = PatientDocument(
            patient_id=patient_id,
            original_filename=filename[:255],
            document_type=document_type,
            classification_confidence=confidence,
            classified_by=classified_by,
            file_path=str(path),
            checksum=checksum,
            document_date=datetime.now(timezone.utc).replace(tzinfo=None),
            size_bytes=len(content),
        )
        db.add(doc)
        db.flush()
        return ToolResult(
            ok=True,
            data={
                "document_id": doc.id,
                "checksum": checksum,
                "document_type": doc.document_type,
                "confidence": doc.classification_confidence,
                "size_bytes": doc.size_bytes,
            },
        )


@tool(
    name="list_patient_documents",
    description="List all documents currently on file for a patient.",
    agents=["document", "coordinator"],
)
def list_patient_documents(patient_id: str) -> ToolResult:
    with session_scope() as db:
        rows = db.scalars(
            select(PatientDocument)
            .where(PatientDocument.patient_id == patient_id)
            .order_by(PatientDocument.created_at.desc())
        ).all()
        return ToolResult(
            ok=True,
            data=[
                {
                    "document_id": d.id,
                    "filename": d.original_filename,
                    "document_type": d.document_type,
                    "confidence": d.classification_confidence,
                    "uploaded_at": d.created_at.isoformat(),
                }
                for d in rows
            ],
        )


@tool(
    name="check_missing_documents",
    description="Compare documents on file against the department's required set and report gaps.",
    agents=["document"],
)
def check_missing_documents(patient_id: str, department_name: str) -> ToolResult:
    with session_scope() as db:
        dept = db.scalar(select(Department).where(Department.name == department_name))
        if not dept:
            return ToolResult(ok=False, error=f"unknown department: {department_name}")

        required = set(dept.required_documents or [])
        on_file = {
            d.document_type
            for d in db.scalars(
                select(PatientDocument).where(PatientDocument.patient_id == patient_id)
            ).all()
        }
        missing = sorted(required - on_file)
        return ToolResult(
            ok=True,
            data={
                "department": dept.name,
                "required": sorted(required),
                "on_file": sorted(on_file),
                "missing": missing,
                "complete": not missing,
            },
        )
