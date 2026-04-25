"""
Shared pytest fixtures.

We use a tiny FakeLLM that mimics the LangChain ChatModel surface area we
actually call — ``invoke`` and ``bind_tools`` — so tests can run without any
real API keys (Gap #1).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, List

import pytest
from langchain_core.messages import AIMessage

# Make `src.*` imports resolve from repo root in CI.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeLLM:
    """
    Minimal stand-in for a LangChain ChatModel. Returns canned ``AIMessage``
    responses in order. ``bind_tools`` returns ``self`` so the reviewer's
    tool-binding path is exercised.
    """

    def __init__(self, responses: Iterable[str] = (), supports_tools: bool = True):
        self._responses: List[str] = list(responses) or [
            "### Executive Summary\nFake review.\n"
        ]
        self._idx = 0
        self.supports_tools = supports_tools
        self.invocations: List[Any] = []

    def invoke(self, messages, **kwargs):
        self.invocations.append(messages)
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            content = self._responses[-1]
        return AIMessage(content=content)

    def bind_tools(self, tools):
        if not self.supports_tools:
            raise RuntimeError("Model does not support bind_tools (test fixture).")
        return self


@pytest.fixture
def fake_llm():
    return FakeLLM(
        responses=[
            "### 1. Feature Summary\nA fake analyzed feature.\n",
            "### Test Suite Overview\nFake suite.\n#### TC-001: Sample\n",
            "### Executive Summary\nFake reviewed.\n",
        ]
    )


@pytest.fixture
def fake_llm_no_tools():
    return FakeLLM(supports_tools=False)
