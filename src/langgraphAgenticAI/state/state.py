from typing import Annotated, Dict

from langgraph.graph.message import add_messages
from typing_extensions import List, TypedDict


class State(TypedDict, total=False):
    """
    Shared state for the QA Intelligence Suite multi-agent workflow
    (RequirementAnalyzer -> TestCaseGenerator -> TestReviewer -> save_report).

    The reviewer runs an autonomous LLM-with-bound-tools loop on a SEPARATE
    message channel (`reviewer_messages`) so its tool-calling history is
    isolated from the rest of the application state.
    """

    # Workflow inputs / outputs (each agent reads earlier outputs and writes its own)
    requirement: str                              # Raw user story / requirement (input)
    analysis: str                                 # Agent 1 output: structured requirement analysis (markdown)
    test_cases: str                               # Agent 2 output: generated Gherkin test cases (markdown)
    sources: List[Dict[str, str]]                 # Agent 3 pre-research: stable source ledger (S1..SN)
    reviewer_messages: Annotated[List, add_messages]  # Private channel for Agent 3 tool-loop
    review: str                                   # Agent 3 output: triage + review report (markdown)
    report_path: str                              # Path where the consolidated report was saved
    # --- Operational fields (Gaps #5, #6, #11) ---
    tool_binding_status: str                      # "ok" | "fallback" | "unavailable" - surfaced in UI
    reviewer_tool_calls: int                      # Counter enforcing MAX_REVIEWER_TOOL_CALLS budget
    dropped_citations: List[str]                  # S-IDs the citation sanitizer stripped post-hoc
