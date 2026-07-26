"""Preflight check for your LLM configuration.

    python -m scripts.check_llm

Verifies the provider in your .env is reachable, lists the model IDs it
actually offers right now, and sends one real request through the same
`complete()` path the agents use. Run this before recording a demo — model IDs
change without notice, and a stale LLM_MODEL is the most common cause of an
app that silently runs in degraded mode.
"""
from __future__ import annotations

import sys

from app.agents import llm
from app.config import settings


def main() -> int:
    provider = (settings.llm_provider or "groq").lower()
    spec = llm.active_provider()

    print(f"Provider      : {provider}")
    print(f"SDK           : {spec.sdk}")
    print(f"Endpoint      : {settings.llm_base_url or spec.base_url or '(sdk default)'}")
    print(f"Model         : {settings.llm_model}")
    print(f"JSON mode     : {'native' if spec.supports_json_mode else 'prompt-enforced'}")

    if not llm.provider_is_configured():
        key = (spec.key_setting or "").upper()
        print(f"\nNOT CONFIGURED — set {key} in .env")
        print("The app will still run, but every workflow is flagged degraded=true.")
        return 1

    client = llm._get_client()
    if client is None:
        print("\nClient could not be constructed. Check the SDK is installed.")
        return 1

    # --- list models, where the provider supports it ---
    print("\nModels available to this key:")
    try:
        if spec.sdk == "anthropic":
            models = [m.id for m in client.models.list().data]
        else:
            models = [m.id for m in client.models.list().data]
        for model_id in sorted(models)[:40]:
            marker = "  <-- your LLM_MODEL" if model_id == settings.llm_model else ""
            print(f"  {model_id}{marker}")
        if settings.llm_model not in models:
            print(f"\n  WARNING: '{settings.llm_model}' is not in that list.")
            print("  Copy an exact id from above into LLM_MODEL.")
    except Exception as exc:  # noqa: BLE001
        print(f"  (provider does not support listing: {type(exc).__name__})")

    # --- one real round trip through the agents' own code path ---
    print("\nLive test via complete():")
    response = llm.complete(
        "You are a test harness. Reply with JSON only.",
        'Return exactly {"ok": true}',
        json_mode=True,
    )
    if response.used_fallback:
        print(f"  FAILED — {response.error}")
        print("  Agents will fall back to deterministic logic.")
        return 1

    print(f"  OK — {response.latency_ms}ms, "
          f"{response.prompt_tokens}+{response.completion_tokens} tokens")
    print(f"  Parsed: {response.as_json()}")
    print("\nConfiguration is live. Agents will use the LLM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
