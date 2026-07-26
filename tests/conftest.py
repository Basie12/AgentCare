"""Test configuration.

Two things are enforced here, both deliberately:

1. **Tests use their own database.** A test run must never touch the database
   you demo from. Set before any app import, because the SQLAlchemy engine is
   constructed at import time.

2. **Tests do not call a live LLM.** Every agent has a deterministic fallback,
   and that is what the suite exercises. Reasons:
     * Determinism — a model's phrasing must not decide whether CI is green.
     * Quota — a full run against a live provider burns real tokens, and free
       tiers have daily caps you will want for demos.
     * Honesty — GitHub Actions has no API key, so this *is* the CI environment.

   Tests that need LLM-shaped behaviour patch `complete` directly (see
   `_fake_llm` in test_agentcare.py), which is faster and fully controllable.

To run against the real provider anyway:

    AGENTCARE_TEST_LIVE_LLM=1 pytest -q
"""
from __future__ import annotations

import os
from pathlib import Path

# --- must happen before any `app.*` import ---------------------------------
_TEST_DIR = Path(__file__).resolve().parent.parent / "data"
_TEST_DIR.mkdir(parents=True, exist_ok=True)

os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DIR / 'test_agentcare.db'}"
os.environ["CHECKPOINT_DB_PATH"] = str(_TEST_DIR / "test_checkpoints.sqlite")
os.environ.setdefault("JWT_SECRET", "test-only-secret-not-used-anywhere-real")

import pytest  # noqa: E402

from app.agents import llm  # noqa: E402
from app.config import settings  # noqa: E402

LIVE = os.getenv("AGENTCARE_TEST_LIVE_LLM") == "1"

_KEY_FIELDS = (
    "groq_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "openrouter_api_key",
    "llm_api_key",
)


@pytest.fixture(scope="session", autouse=True)
def _llm_policy():
    """Blank every provider key so `complete()` takes its degraded path."""
    if LIVE:
        yield
        return

    saved = {field: getattr(settings, field, None) for field in _KEY_FIELDS}
    for field in _KEY_FIELDS:
        setattr(settings, field, None)
    llm.reset_client()

    yield

    for field, value in saved.items():
        setattr(settings, field, value)
    llm.reset_client()


@pytest.fixture(scope="session", autouse=True)
def _fresh_test_database(_llm_policy):
    """Start every session from an empty database, then clean up after."""
    from app.db.base import engine, init_db
    from app.db.seed import seed

    db_path = _TEST_DIR / "test_agentcare.db"
    checkpoint_path = _TEST_DIR / "test_checkpoints.sqlite"
    for path in (db_path, checkpoint_path):
        path.unlink(missing_ok=True)

    init_db()
    seed()
    yield

    engine.dispose()
    # SQLite leaves -shm/-wal sidecars behind in WAL mode.
    for path in (db_path, checkpoint_path):
        for suffix in ("", "-shm", "-wal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)


def pytest_report_header(config):
    mode = "LIVE provider" if LIVE else "deterministic fallbacks (no LLM calls)"
    return f"agentcare: {mode}, isolated test database"
