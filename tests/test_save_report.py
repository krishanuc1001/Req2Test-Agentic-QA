"""Tests for save_report file naming and content (Gaps #1, #7)."""
import os
import re
from pathlib import Path

from src.langgraphAgenticAI.nodes.test_reviewer_node import TestReviewerAgent


def test_save_report_uses_uuid_suffix(tmp_path, monkeypatch, fake_llm):
    monkeypatch.chdir(tmp_path)
    agent = TestReviewerAgent(fake_llm)
    state = {
        "requirement": "x",
        "analysis": "a",
        "test_cases": "tc",
        "review": "r",
        "sources": [],
    }
    out = agent.save_report(state)
    p = out["report_path"]
    assert os.path.exists(p)
    name = os.path.basename(p)
    # qa_report_<14-digit-timestamp>_<8-hex>.md
    assert re.fullmatch(r"qa_report_\d{8}_\d{6}_[0-9a-f]{8}\.md", name), name


def test_save_report_two_runs_produce_distinct_files(tmp_path, monkeypatch, fake_llm):
    monkeypatch.chdir(tmp_path)
    agent = TestReviewerAgent(fake_llm)
    state = {"requirement": "x", "analysis": "", "test_cases": "", "review": "", "sources": []}
    p1 = agent.save_report(state)["report_path"]
    p2 = agent.save_report(state)["report_path"]
    assert p1 != p2, "UUID suffix must prevent collisions"


def test_save_report_includes_all_sections(tmp_path, monkeypatch, fake_llm):
    monkeypatch.chdir(tmp_path)
    agent = TestReviewerAgent(fake_llm)
    state = {
        "requirement": "REQ-X",
        "analysis": "AN-X",
        "test_cases": "TC-X",
        "review": "RV-X",
        "sources": [{"id": "S1", "category": "automation", "title": "T1", "url": "https://x"}],
    }
    p = agent.save_report(state)["report_path"]
    content = Path(p).read_text(encoding="utf-8")
    for marker in ("REQ-X", "AN-X", "TC-X", "RV-X", "| S1 |"):
        assert marker in content, f"missing {marker} in saved report"
