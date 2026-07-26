"""LLM access layer — provider agnostic.

Swap providers by editing LLM_PROVIDER in .env. Nothing else in the codebase
changes, because every agent calls `complete()` and never touches a vendor SDK.

Most providers expose an OpenAI-compatible endpoint, so they share one code
path with a different base_url. Anthropic uses its own SDK.

Three guarantees the rest of the codebase depends on:
  1. Text is PII-redacted before it leaves the process.
  2. Calls retry with backoff, then degrade — this function never raises.
  3. Every call reports whether it fell back, so the workflow can mark itself
     `degraded` instead of silently pretending an agent ran.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.safety.pii import redact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderSpec:
    sdk: str                      # "openai" (compatible) or "anthropic"
    base_url: str | None
    key_setting: str | None       # attribute on Settings holding the API key
    supports_json_mode: bool = True
    requires_key: bool = True


PROVIDERS: dict[str, ProviderSpec] = {
    # Free tier, very fast. Default.
    "groq": ProviderSpec("openai", "https://api.groq.com/openai/v1", "groq_api_key"),
    "openai": ProviderSpec("openai", None, "openai_api_key"),
    "anthropic": ProviderSpec("anthropic", None, "anthropic_api_key", supports_json_mode=False),
    "gemini": ProviderSpec(
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "google_api_key",
    ),
    # One key, many models. Handy for A/B testing models during the build.
    "openrouter": ProviderSpec("openai", "https://openrouter.ai/api/v1", "openrouter_api_key"),
    # Fully local, no key, no rate limits, works offline.
    "ollama": ProviderSpec("openai", "http://localhost:11434/v1", None, requires_key=False),
    # Anything else OpenAI-compatible: set LLM_BASE_URL and LLM_API_KEY yourself.
    "custom": ProviderSpec("openai", None, "llm_api_key"),
}


def active_provider() -> ProviderSpec:
    name = (settings.llm_provider or "groq").lower()
    if name not in PROVIDERS:
        logger.warning("unknown LLM_PROVIDER '%s', falling back to groq", name)
        return PROVIDERS["groq"]
    return PROVIDERS[name]


def provider_api_key() -> str | None:
    spec = active_provider()
    if spec.key_setting is None:
        return None
    return getattr(settings, spec.key_setting, None)


def provider_is_configured() -> bool:
    spec = active_provider()
    return True if not spec.requires_key else bool(provider_api_key())


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    used_fallback: bool = False
    error: str | None = None
    redaction_hits: list[str] = field(default_factory=list)

    def as_json(self, default: dict | None = None) -> dict:
        """Parse a JSON object out of the response, tolerating code fences."""
        if not self.text:
            return default or {}
        cleaned = re.sub(r"^```(?:json)?|```$", "", self.text.strip(), flags=re.M).strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return default or {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("LLM returned unparseable JSON")
            return default or {}


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------
_client: Any | None = None
_client_key: str | None = None


def _get_client() -> Any | None:
    """Cached per provider, so switching LLM_PROVIDER rebuilds it."""
    global _client, _client_key

    provider_name = (settings.llm_provider or "groq").lower()
    if _client is not None and _client_key == provider_name:
        return _client

    spec = active_provider()
    if not provider_is_configured():
        return None

    api_key = provider_api_key()
    base_url = settings.llm_base_url or spec.base_url

    try:
        if spec.sdk == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key, timeout=settings.llm_timeout_seconds)
        else:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key or "not-needed",   # Ollama ignores this
                base_url=base_url,
                timeout=settings.llm_timeout_seconds,
            )
    except Exception:  # noqa: BLE001
        logger.exception("could not construct client for provider '%s'", provider_name)
        return None

    _client, _client_key = client, provider_name
    return _client


def reset_client() -> None:
    """Force the next call to rebuild the client. Used by tests."""
    global _client, _client_key
    _client, _client_key = None, None


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _call_openai_compatible(client: Any, system: str, user: str, json_mode: bool) -> tuple:
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    if json_mode and active_provider().supports_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    usage = getattr(completion, "usage", None)
    return (
        (completion.choices[0].message.content or "").strip(),
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _call_anthropic(client: Any, system: str, user: str, json_mode: bool) -> tuple:
    # Anthropic has no response_format flag, so JSON is requested in the system
    # prompt and enforced by our own parser.
    if json_mode:
        system = f"{system}\n\nRespond with a single valid JSON object and nothing else."

    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()
    usage = getattr(message, "usage", None)
    return (
        text,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


def complete(
    system_prompt: str,
    user_content: str,
    *,
    json_mode: bool = False,
    redact_pii: bool = True,
) -> LLMResponse:
    """Never raises. On failure returns used_fallback=True with an empty body."""
    started = time.perf_counter()
    provider_name = (settings.llm_provider or "groq").lower()

    redaction = redact(user_content) if redact_pii else None
    payload = redaction.text if redaction else user_content
    hits = redaction.hits if redaction else []

    def _failed(error: str) -> LLMResponse:
        return LLMResponse(
            text="",
            provider=provider_name,
            used_fallback=True,
            error=error,
            latency_ms=int((time.perf_counter() - started) * 1000),
            redaction_hits=hits,
        )

    client = _get_client()
    if client is None:
        spec = active_provider()
        key_name = (spec.key_setting or "").upper() or "no key required"
        return _failed(f"LLM not configured (provider={provider_name}, expected {key_name})")

    caller = (
        _call_anthropic if active_provider().sdk == "anthropic" else _call_openai_compatible
    )

    try:
        text, prompt_tokens, completion_tokens = caller(
            client, system_prompt, payload, json_mode
        )
    except (RetryError, Exception) as exc:  # noqa: BLE001
        logger.warning("LLM call failed after retries (%s): %s", provider_name, exc)
        return _failed(f"{type(exc).__name__}: {exc}")

    return LLMResponse(
        text=text,
        model=settings.llm_model,
        provider=provider_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        used_fallback=False,
        redaction_hits=hits,
    )
