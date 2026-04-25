import logging
import os

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraphAgenticAI.nodes.requirement_analyzer_node import RequirementAnalyzerAgent
from src.langgraphAgenticAI.nodes.test_case_generator_node import TestCaseGeneratorAgent
from src.langgraphAgenticAI.nodes.test_reviewer_node import TestReviewerAgent
from src.langgraphAgenticAI.state.state import State

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Constructs the QA Intelligence Suite multi-agent LangGraph workflow.

    Pipeline:

        START
          -> requirement_analyzer  (Agent 1: Senior BA / QA Lead)
          -> test_case_generator   (Agent 2: Senior SDET - Gherkin output)
          -> reviewer_research     (Agent 3 pre-fetch: 3 targeted Tavily queries
                                    building a stable S1..SN source ledger)
          -> reviewer_init         (primes a private `reviewer_messages` channel
                                    with system prompt + citation rules + sources legend)
          -> reviewer_agent        (LLM with `bind_tools([Tavily])`)
          -> [tools_condition on reviewer_messages]
                   "tools" -> reviewer_tools -> reviewer_agent  (autonomous loop)
                   END     -> reviewer_finalize
          -> reviewer_finalize     (extracts final AIMessage into state.review)
          -> save_report           (writes QAReports/qa_report_<timestamp>.md)
        END

    Graceful degradation: when `TAVILY_API_KEY` is not set, `reviewer_research`
    returns an empty source ledger and the tool-loop wiring is omitted -- the
    Reviewer runs as a single LLM step with no external grounding.
    """

    USECASE = "QA Intelligence Suite"

    def __init__(self, model):
        """
        ``model`` may be either:
          * a single LLM client (back-compat: same model for all 3 agents), or
          * a dict mapping agent role -> LLM client, e.g.::

                {"analyzer": fast_llm, "generator": fast_llm, "reviewer": strong_llm}

        Per-agent assignment lets the cheap Analyzer/Generator share an
        instant-speed model while the Reviewer (which uses tools and reasoning)
        runs on a stronger one (Gap #10). Keys may be omitted; a missing role
        falls back to ``model["default"]`` or the first value in the dict.
        """
        if isinstance(model, dict):
            self._models = model
            default = model.get("default") or next(iter(model.values()))
            self.llm = default
        else:
            self._models = {"default": model}
            self.llm = model
        self.graph_builder = StateGraph(State)

    def _model_for(self, role: str):
        """Return the LLM bound to ``role``, falling back to the default."""
        return self._models.get(role) or self._models.get("default") or self.llm

    def setup_graph(self, usecase: str):
        """
        Builds and compiles the QA Intelligence Suite graph. Raises
        ValueError for any other use case to keep configuration honest.

        Optional checkpointing (Gap #3): when env var
        ``LANGGRAPH_CHECKPOINT_ENABLED`` is truthy, an in-memory ``MemorySaver``
        is wired so callers passing ``config={"configurable": {"thread_id": X}}``
        can resume runs and inspect intermediate state.
        """
        if usecase != self.USECASE:
            raise ValueError(
                f"Unsupported use case: {usecase!r}. "
                f"Only {self.USECASE!r} is supported."
            )
        self.qa_intelligence_graph()

        checkpointer = None
        if os.environ.get("LANGGRAPH_CHECKPOINT_ENABLED", "").lower() in ("1", "true", "yes"):
            try:
                from langgraph.checkpoint.memory import MemorySaver
                checkpointer = MemorySaver()
                logger.info("LangGraph MemorySaver checkpointer enabled.")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to enable checkpointer: %s", exc)

        if checkpointer is not None:
            return self.graph_builder.compile(checkpointer=checkpointer)
        return self.graph_builder.compile()

    def qa_intelligence_graph(self):
        analyzer_agent = RequirementAnalyzerAgent(self._model_for("analyzer"))
        generator_agent = TestCaseGeneratorAgent(self._model_for("generator"))
        reviewer_agent = TestReviewerAgent(self._model_for("reviewer"))

        # Linear agent nodes
        self.graph_builder.add_node("requirement_analyzer", analyzer_agent.analyze)
        self.graph_builder.add_node("test_case_generator", generator_agent.generate)
        self.graph_builder.add_node("reviewer_research", reviewer_agent.research)
        self.graph_builder.add_node("reviewer_init", reviewer_agent.init_messages)
        self.graph_builder.add_node("reviewer_agent", reviewer_agent.agent)
        self.graph_builder.add_node("reviewer_finalize", reviewer_agent.finalize)
        self.graph_builder.add_node("save_report", reviewer_agent.save_report)

        # Linear edges up to the reviewer LLM
        self.graph_builder.set_entry_point("requirement_analyzer")
        self.graph_builder.add_edge("requirement_analyzer", "test_case_generator")
        self.graph_builder.add_edge("test_case_generator", "reviewer_research")
        self.graph_builder.add_edge("reviewer_research", "reviewer_init")
        self.graph_builder.add_edge("reviewer_init", "reviewer_agent")

        # Autonomous tool loop only when Tavily is bound; degrade gracefully otherwise.
        if reviewer_agent.has_tavily and reviewer_agent.tavily_tool is not None:
            self.graph_builder.add_node(
                "reviewer_tools",
                ToolNode(
                    tools=[reviewer_agent.tavily_tool],
                    messages_key="reviewer_messages",
                ),
            )

            def reviewer_tools_condition(state):
                # Routes to "tools" if the latest reviewer AIMessage has tool_calls,
                # otherwise to END (which we redirect to reviewer_finalize below).
                return tools_condition(state, messages_key="reviewer_messages")

            self.graph_builder.add_conditional_edges(
                "reviewer_agent",
                reviewer_tools_condition,
                {"tools": "reviewer_tools", END: "reviewer_finalize"},
            )
            self.graph_builder.add_edge("reviewer_tools", "reviewer_agent")
        else:
            # No Tavily -> single-shot reviewer, no tool node.
            self.graph_builder.add_edge("reviewer_agent", "reviewer_finalize")

        # Terminal segment
        self.graph_builder.add_edge("reviewer_finalize", "save_report")
        self.graph_builder.add_edge("save_report", END)
