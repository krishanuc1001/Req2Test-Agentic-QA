"""Tests for tool-call budget + bind_tools status (Gaps #6, #11)."""
from langchain_core.messages import AIMessage, SystemMessage

from src.langgraphAgenticAI.nodes.test_reviewer_node import (
    MAX_REVIEWER_TOOL_CALLS,
    TestReviewerAgent,
)


def test_bind_tools_status_unavailable_without_tavily(monkeypatch, fake_llm):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = TestReviewerAgent(fake_llm)
    assert agent.bind_tools_status == "unavailable"


def test_bind_tools_status_fallback_when_provider_rejects(monkeypatch, fake_llm_no_tools):
    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    # Force _build_tavily_tool to return a sentinel object so __init__
    # tries bind_tools and the fixture raises -> "fallback" path.
    sentinel = object()
    monkeypatch.setattr(
        TestReviewerAgent, "_build_tavily_tool", staticmethod(lambda: sentinel)
    )
    agent = TestReviewerAgent(fake_llm_no_tools)
    assert agent.bind_tools_status == "fallback"


def test_init_messages_seeds_counters_and_status(fake_llm, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = TestReviewerAgent(fake_llm)
    out = agent.init_messages({"analysis": "A", "test_cases": "TC", "sources": []})
    assert out["reviewer_tool_calls"] == 0
    assert out["tool_binding_status"] == "unavailable"
    assert len(out["reviewer_messages"]) == 2  # system + user


def test_agent_blocks_further_tool_calls_when_budget_exhausted(fake_llm, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = TestReviewerAgent(fake_llm)
    state = {
        "reviewer_messages": [SystemMessage(content="sys")],
        "reviewer_tool_calls": MAX_REVIEWER_TOOL_CALLS,
    }
    out = agent.agent(state)
    # Plain LLM (without tools) gets used; response is an AIMessage.
    assert isinstance(out["reviewer_messages"][0], AIMessage)
