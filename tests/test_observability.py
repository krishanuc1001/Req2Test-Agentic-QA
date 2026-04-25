"""Tests for the observability bootstrap (Gap #9)."""
import logging
import os

from src.langgraphAgenticAI.observability.setup import (
    get_token_counter_callback,
    setup_observability,
)


def test_setup_observability_is_idempotent(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    setup_observability()
    setup_observability()  # second call must not raise
    assert logging.getLogger().handlers, "stdlib logging must be configured"


def test_setup_observability_enables_langsmith_when_key_present(monkeypatch):
    # Reset internal flag so we can re-run setup in this test process.
    import src.langgraphAgenticAI.observability.setup as mod
    monkeypatch.setattr(mod, "_DONE", False)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    setup_observability()

    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_PROJECT") == "qa-intelligence-suite"


def test_token_counter_callback_returns_object_or_none():
    cb = get_token_counter_callback()
    # Must either be a usable handler or gracefully None.
    if cb is not None:
        assert hasattr(cb, "totals")
        assert cb.totals == {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }
