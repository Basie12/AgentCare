"""Environment-based configuration. No secrets are ever hardcoded."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Application ---
    app_name: str = "AgentCare"
    environment: str = "development"
    debug: bool = True

    # --- Database ---
    # SQLite by default so the project runs with zero setup.
    # Set DATABASE_URL=postgresql+psycopg://... for Postgres.
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'agentcare.db'}"

    # --- LangGraph checkpointer (persistent agent state) ---
    checkpoint_db_path: str = str(BASE_DIR / "data" / "checkpoints.sqlite")

    # --- LLM ---
    # Which provider to use. See PROVIDERS in app/agents/llm.py.
    # groq | openai | anthropic | gemini | openrouter | ollama | custom
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 30

    # Only the key for your chosen provider needs a value.
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    openrouter_api_key: str | None = None
    llm_api_key: str | None = None      # provider "custom"
    llm_base_url: str | None = None     # override the provider default endpoint

    # --- Auth ---
    jwt_secret: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720

    # --- Documents ---
    document_storage_dir: str = str(BASE_DIR / "storage" / "documents")
    max_document_mb: int = 10

    @property
    def llm_enabled(self) -> bool:
        """When False the system runs on deterministic fallbacks (degraded mode)."""
        from app.agents.llm import provider_is_configured

        return provider_is_configured()


    def ensure_directories(self) -> None:
        """Create runtime directories if missing.

        A fresh clone has no data/ or storage/ (their contents are gitignored),
        and SQLite will not create a missing parent directory — it just fails
        with "unable to open database file". Belt and braces alongside the
        committed .gitkeep files.
        """
        targets = [
            Path(self.document_storage_dir),
            Path(self.checkpoint_db_path).parent,
        ]

        url = self.database_url
        if url.startswith("sqlite"):
            raw = url.split("///", 1)[-1]
            if raw and raw != ":memory:":
                targets.append(Path(raw).expanduser().resolve().parent)

        for target in targets:
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass


@lru_cache
def get_settings() -> Settings:
    instance = Settings()
    instance.ensure_directories()
    return instance


settings = get_settings()
