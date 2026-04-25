"""Tests for reviewer finalize handling of provider response shapes (Gap #1)."""
from langchain_core.messages import AIMessage, HumanMessage

from src.langgraphAgenticAI.nodes.test_reviewer_node import TestReviewerAgent


def test_finalize_extracts_string_content():
    state = {
        "reviewer_messages": [
            HumanMessage(content="ignored"),
            AIMessage(content="### Executive Summary\nAll good.\n"),
        ],
        "sources": [],
    }
    out = TestReviewerAgent.finalize(state)
    assert "Executive Summary" in out["review"]
    assert out["dropped_citations"] == []


def test_finalize_extracts_list_of_parts_content():
    """Gemini-style content: list of {'text': '...'} dicts."""
    state = {
        "reviewer_messages": [
            AIMessage(content=[{"text": "Part 1. "}, {"text": "Part 2."}]),
        ],
        "sources": [{"id": "S1"}],
    }
    out = TestReviewerAgent.finalize(state)
    assert out["review"].startswith("Part 1.")
    assert "Part 2." in out["review"]


def test_finalize_drops_hallucinated_citations():
    state = {
        "reviewer_messages": [
            AIMessage(content="Use Playwright (ref: S1, S99) for smoke."),
        ],
        "sources": [{"id": "S1", "title": "real"}],
    }
    out = TestReviewerAgent.finalize(state)
    assert "S99" not in out["review"]
    assert "S99" in out["dropped_citations"]
    assert "S1" in out["review"]


def test_finalize_handles_empty_history():
    out = TestReviewerAgent.finalize({"reviewer_messages": [], "sources": []})
    assert "no textual output" in out["review"].lower()
