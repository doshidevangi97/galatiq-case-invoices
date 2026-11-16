"""Shared Groq helpers: client creation and token-usage recording."""

import logging
import os
import time

from ingestion.models import IngestionError, LLMCall, LLMUsage

DEFAULT_MODEL = "openai/gpt-oss-120b"


def default_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def has_api_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def make_client():
    if not has_api_key():
        raise IngestionError(
            "GROQ_API_KEY is not set. It is needed for LLM steps. Copy .env.example to .env and add your key."
        )
    from groq import Groq

    # Free-tier accounts have low tokens-per-minute limits; the SDK waits and retries on 429s.
    return Groq(max_retries=5)


def record_call(usage: LLMUsage, response, *, label: str, attempt: int, model: str, started: float,
                log: logging.Logger) -> LLMCall:
    """Append one call's token counts (from the Groq response) to `usage` and log them."""
    raw = getattr(response, "usage", None)
    details = getattr(raw, "completion_tokens_details", None)
    call = LLMCall(
        file=label, attempt=attempt, model=model,
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    usage.calls.append(call)
    log.info("LLM call for %s: %d tokens (%d prompt + %d completion) in %d ms",
             label, call.total_tokens, call.prompt_tokens, call.completion_tokens, call.latency_ms,
             extra={"data": call.model_dump()})
    return call
