"""
Pydantic schemas for structured agent output (Gap #2 scaffold).

These models describe the *target* JSON shape we want each agent to emit
once the project migrates from Markdown-only to ``with_structured_output``.
They are defined here so that:

1. Future migration is a small, well-isolated change in each node:
       parsed = self.llm.with_structured_output(RequirementAnalysis).invoke(...)
2. Unit tests can assert the schema is stable and import-safe today.
3. Downstream consumers (eval harness, exporters) can already start
   parsing Markdown into these objects.

Marked as scaffold: not yet wired into runtime to keep this PR
non-breaking.
"""
from __future__ import annotations

from typing import List, Literal, Optional

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - pydantic is a hard dep but be defensive
    raise


Priority = Literal["P1", "P2", "P3", "P4"]
RiskLevel = Literal["High", "Medium", "Low"]
TestCategory = Literal[
    "Positive", "Negative", "Boundary", "Security", "Accessibility", "Performance"
]


class RequirementAnalysis(BaseModel):
    """Agent 1 output."""

    feature_summary: str
    actors: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    nfrs: List[str] = Field(default_factory=list)


class TestCase(BaseModel):
    """One generated Gherkin test case."""

    id: str = Field(description="Sequential ID like TC-001")
    title: str
    category: TestCategory
    preconditions: str = ""
    test_data: str = ""
    gherkin: str


class TestSuite(BaseModel):
    """Agent 2 output."""

    overview: str
    test_cases: List[TestCase] = Field(default_factory=list)


class TriageRow(BaseModel):
    test_id: str
    priority: Priority
    risk: RiskLevel
    automation_feasibility: RiskLevel  # High/Medium/Low repurposed
    recommended_tool: str
    rationale: str
    citations: List[str] = Field(default_factory=list)


class ReviewReport(BaseModel):
    """Agent 3 output."""

    executive_summary: str
    triage: List[TriageRow] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(default_factory=list)
    risk_assessment: List[str] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
