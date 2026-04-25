"""Smoke tests: graph compiles + state shape (Gap #1)."""
from src.langgraphAgenticAI.graph.graph_builder import GraphBuilder
from src.langgraphAgenticAI.state.state import State


def test_graph_compiles_with_single_llm(fake_llm, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    graph = GraphBuilder(model=fake_llm).setup_graph(GraphBuilder.USECASE)
    assert graph is not None


def test_graph_compiles_with_per_agent_models(fake_llm, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    models = {"analyzer": fake_llm, "generator": fake_llm, "reviewer": fake_llm}
    graph = GraphBuilder(model=models).setup_graph(GraphBuilder.USECASE)
    assert graph is not None


def test_setup_graph_rejects_unknown_usecase(fake_llm):
    import pytest
    builder = GraphBuilder(model=fake_llm)
    with pytest.raises(ValueError):
        builder.setup_graph("Basic Chatbot")


def test_state_typed_dict_keys_present():
    annotations = State.__annotations__
    for key in (
        "requirement", "analysis", "test_cases", "sources",
        "reviewer_messages", "review", "report_path",
        "tool_binding_status", "reviewer_tool_calls", "dropped_citations",
    ):
        assert key in annotations, f"missing State key: {key}"
