"""Synthetic seed data.

Everything here is fabricated with a fixed random seed. No real patient data,
no real credentials. Run with:  python -m app.db.seed
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from faker import Faker
from sqlalchemy import select

from app.auth.security import hash_password
from app.db.base import init_db, session_scope
from app.db.models import (
    AppointmentSlot,
    Department,
    Doctor,
    PatientProfile,
    Role,
    SlotStatus,
    User,
)

fake = Faker()
Faker.seed(42)
random.seed(42)

DEPARTMENTS = [
    ("Cardiology", "Heart and vascular appointments, ECG review visits, cardiac follow-ups.",
     ["ecg", "referral_letter"]),
    ("Orthopaedics", "Bone, joint and fracture clinics, post-surgical reviews, physiotherapy referrals.",
     ["imaging_report", "referral_letter"]),
    ("Dermatology", "Skin clinic appointments, mole checks, chronic skin condition reviews.",
     ["referral_letter"]),
    ("General Medicine", "General adult outpatient consultations and routine reviews.",
     ["identity_proof"]),
    ("Paediatrics", "Child and adolescent outpatient appointments and routine checks.",
     ["identity_proof", "consent_form"]),
    ("Radiology", "Imaging appointments including X-ray, ultrasound, CT and MRI scheduling.",
     ["referral_letter"]),
    ("Ophthalmology", "Eye clinic appointments, vision assessments and post-operative reviews.",
     ["referral_letter"]),
    ("ENT", "Ear, nose and throat clinic appointments and hearing assessments.",
     ["referral_letter"]),
    ("General Surgery",
     "Pre-operative assessment, day-surgery scheduling, post-surgical review "
     "appointments and surgical outpatient clinics.",
     ["referral_letter", "consent_form", "identity_proof", "imaging_report"]),
]

DEMO_USERS = [
    ("Asha Patel", "patient@demo.local", "patient123", Role.PATIENT),
    ("Daniel Okafor", "patient2@demo.local", "patient123", Role.PATIENT),
    ("Reception Desk", "staff@demo.local", "staff123", Role.STAFF),
    ("System Admin", "admin@demo.local", "admin123", Role.ADMIN),
]


def seed() -> None:
    init_db()

    with session_scope() as db:
        if db.scalar(select(Department).limit(1)):
            return _report("Database already seeded — skipping.")

        # Departments and doctors
        doctors: list[Doctor] = []
        for name, description, required in DEPARTMENTS:
            dept = Department(
                name=name, description=description, required_documents=required, active=True
            )
            db.add(dept)
            db.flush()
            for _ in range(random.randint(2, 3)):
                doctor = Doctor(
                    department_id=dept.id, name=f"Dr. {fake.last_name()}", active=True
                )
                db.add(doctor)
                doctors.append(doctor)
        db.flush()

        # Two weeks of 30-minute slots, weekdays 09:00-16:00
        base = datetime.now(timezone.utc).replace(
            hour=9, minute=0, second=0, microsecond=0, tzinfo=None
        )
        slot_count = 0
        for doctor in doctors:
            for day_offset in range(1, 15):
                day = base + timedelta(days=day_offset)
                if day.weekday() >= 5:
                    continue
                for block in range(14):  # 09:00 -> 16:00
                    start = day + timedelta(minutes=30 * block)
                    if random.random() < 0.45:  # not every slot is offered
                        continue
                    db.add(
                        AppointmentSlot(
                            doctor_id=doctor.id,
                            start_time=start,
                            end_time=start + timedelta(minutes=30),
                            status=SlotStatus.OPEN,
                        )
                    )
                    slot_count += 1
        db.flush()

        # Demo accounts
        for name, email, password, role in DEMO_USERS:
            user = User(
                name=name, email=email, password_hash=hash_password(password), role=role
            )
            db.add(user)
            db.flush()
            if role is Role.PATIENT:
                db.add(
                    PatientProfile(
                        user_id=user.id,
                        date_of_birth=fake.date_of_birth(minimum_age=22, maximum_age=78).isoformat(),
                        phone=fake.numerify("###-555-####"),
                        preferred_language="en",
                        emergency_contact=f"{fake.name()} ({fake.numerify('###-555-####')})",
                    )
                )

        summary = (
            f"Seeded {len(DEPARTMENTS)} departments, {len(doctors)} doctors, "
            f"{slot_count} slots."
        )

    # Reporting happens *outside* the transaction. Printing inside it means a
    # closed stdout (e.g. `python -m app.db.seed | head -2`) raises
    # BrokenPipeError mid-transaction and silently rolls back the whole seed.
    _report(summary)


def _report(summary: str) -> None:
    try:
        print(summary)
        print("\nDemo logins:")
        for name, email, password, role in DEMO_USERS:
            print(f"  {role.value:<8} {email:<22} {password}")
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    seed()