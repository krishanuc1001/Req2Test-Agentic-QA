"""Lightweight schema validation (Gap #2 scaffold)."""
import pytest

from src.langgraphAgenticAI.schemas import (
    RequirementAnalysis,
    ReviewReport,
    TestCase,
    TestSuite,
    TriageRow,
)


def test_requirement_analysis_minimal():
    obj = RequirementAnalysis(feature_summary="x")
    assert obj.feature_summary == "x"
    assert obj.actors == []


def test_test_case_category_enforced():
    with pytest.raises(Exception):
        TestCase(id="TC-001", title="t", category="Random", gherkin="Scenario: x")


def test_triage_row_priority_enforced():
    with pytest.raises(Exception):
        TriageRow(
            test_id="TC-001", priority="P9", risk="High",
            automation_feasibility="High", recommended_tool="Playwright",
            rationale="x",
        )


def test_review_report_round_trip():
    rep = ReviewReport(
        executive_summary="ok",
        triage=[
            TriageRow(
                test_id="TC-001", priority="P1", risk="High",
                automation_feasibility="High", recommended_tool="Playwright",
                rationale="smoke", citations=["S1"],
            )
        ],
    )
    payload = rep.model_dump()
    assert payload["triage"][0]["test_id"] == "TC-001"


def test_test_suite_default_empty():
    suite = TestSuite(overview="empty")
    assert suite.test_cases == []
