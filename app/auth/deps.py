"""Role-based access control, enforced in the backend.

Hiding a button is not access control. Every protected route depends on one of
these, and patient-scoped resources additionally verify ownership so patient A
cannot read patient B's records by guessing an id.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.db.base import get_db
from app.db.models import PatientProfile, Role, User


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("access_token")
    if not token:
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            token = header[7:]

    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session")

    user = db.get(User, payload.get("sub"))
    if not user or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user


def require_roles(*roles: Role):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires role: {', '.join(r.value for r in roles)}",
            )
        return user

    return dependency


require_patient = require_roles(Role.PATIENT)
require_staff = require_roles(Role.STAFF, Role.ADMIN)
require_admin = require_roles(Role.ADMIN)


def current_patient_profile(
    user: User = Depends(require_patient), db: Session = Depends(get_db)
) -> PatientProfile:
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == user.id).first()
    if not profile:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No patient profile for this account")
    return profile


def assert_owns_patient(user: User, patient_id: str, db: Session) -> None:
    """Staff may read any patient. A patient may only read their own record."""
    if user.role in (Role.STAFF, Role.ADMIN):
        return
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == user.id).first()
    if not profile or profile.id != patient_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not permitted to access this record")
