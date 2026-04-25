import os
import uuid
import streamlit as st


# Per-node user-friendly status labels for the streaming progress UX (Gap #14).
NODE_STATUS_LABELS = {
    "requirement_analyzer": "🧠 Agent 1 · Analyzing requirement...",
    "test_case_generator":  "🧪 Agent 2 · Generating Gherkin test cases...",
    "reviewer_research":    "🔎 Agent 3 · Researching automation/security/NFR sources...",
    "reviewer_init":        "🧷 Agent 3 · Priming reviewer context...",
    "reviewer_agent":       "🤖 Agent 3 · Reasoning (LLM call)...",
    "reviewer_tools":       "🛠️ Agent 3 · Executing Tavily tool call...",
    "reviewer_finalize":    "📝 Agent 3 · Finalizing review...",
    "save_report":          "💾 Saving consolidated report...",
}


class DisplayResultStreamlit:
    """
    Renders the QA Intelligence Suite multi-agent workflow output in Streamlit:
      - Tab 1: Agent 1 - Requirement Analysis
      - Tab 2: Agent 2 - Generated Test Cases (Gherkin)
      - Tab 3: Agent 3 - Review & Triage Report
      - Tab 4: Sources (audit trail of pre-researched references S1..SN)
      - Tab 5: Consolidated Markdown Report (with download button)

    Streams the graph (Gap #14) so users see per-agent progress instead of
    a 30-60s blank spinner. Surfaces ``tool_binding_status`` as a UI banner
    (Gap #11) and any dropped citations (Gap #5).
    """

    # Gap #6: cap the LangGraph recursion budget so a misbehaving model
    # cannot run away with the graph (default is 25; 12 is plenty given the
    # MAX_REVIEWER_TOOL_CALLS=4 budget upstream).
    RECURSION_LIMIT = 12

    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        requirement = self.user_message
        if not requirement or not requirement.strip():
            st.info(
                "Paste a user story / requirement in the sidebar and click "
                "**Run Multi-Agent QA Workflow**."
            )
            return

        # Show the user's requirement for context
        with st.chat_message("user"):
            st.markdown("**📝 Requirement submitted:**")
            st.markdown(requirement)

        # Stable thread_id so an optional checkpointer (Gap #3) can resume.
        if "qa_thread_id" not in st.session_state:
            st.session_state.qa_thread_id = uuid.uuid4().hex
        run_config = {
            "recursion_limit": self.RECURSION_LIMIT,
            "configurable": {"thread_id": st.session_state.qa_thread_id},
        }

        progress = st.empty()
        result = {}
        try:
            # Streaming mode 'updates' yields {node_name: partial_state} per step.
            for chunk in self.graph.stream(
                {"requirement": requirement},
                config=run_config,
                stream_mode="updates",
            ):
                for node_name, partial in chunk.items():
                    label = NODE_STATUS_LABELS.get(node_name, f"⏳ {node_name}...")
                    progress.info(label)
                    if isinstance(partial, dict):
                        result.update(partial)
            progress.empty()
        except Exception as e:
            progress.empty()
            st.error(f"Error: Multi-agent workflow failed: {e}")
            return

        analysis_md = result.get("analysis", "_No analysis produced._")
        test_cases_md = result.get("test_cases", "_No test cases produced._")
        review_md = result.get("review", "_No review produced._")
        report_path = result.get("report_path")
        sources = result.get("sources") or []
        tool_binding_status = result.get("tool_binding_status", "unavailable")
        dropped_citations = result.get("dropped_citations") or []

        st.success("✅ Multi-agent workflow completed.")

        # Gap #11: surface bind_tools fallback so reviewer-loss isn't silent.
        if tool_binding_status == "fallback":
            st.warning(
                "⚠️ The selected model did not accept `bind_tools`. Agent 3 "
                "ran without tool access — its review is based solely on the "
                "pre-researched source ledger. Try a model with native tool "
                "support (e.g. `llama-3.3-70b-versatile` or `gemini-2.5-pro`)."
            )
        elif tool_binding_status == "unavailable":
            st.info(
                "ℹ️ Agent 3 ran without web grounding (Tavily key not set). "
                "Add a Tavily API key in the sidebar for source-cited reviews."
            )

        # Gap #5: tell the user when we stripped hallucinated S-IDs.
        if dropped_citations:
            st.warning(
                "🧹 Stripped hallucinated source citations from reviewer "
                f"output: {', '.join(dropped_citations)}. The model invented "
                "these IDs; they are not in the source ledger."
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧠 Agent 1 · Analysis",
            "🧪 Agent 2 · Test Cases",
            "🔍 Agent 3 · Review & Triage",
            "📚 Sources (Audit Trail)",
            "📄 Consolidated Report",
        ])

        with tab1:
            st.markdown("#### Requirement Analyzer (Senior BA / QA Lead)")
            st.markdown(analysis_md)

        with tab2:
            st.markdown("#### Test Case Generator (Senior SDET)")
            st.markdown(test_cases_md)

        with tab3:
            st.markdown("#### Test Reviewer & Triager (QA Manager)")
            st.markdown(review_md)

        with tab4:
            st.markdown("#### Pre-Researched Source Ledger")
            st.caption(
                "These sources are referenced by the Reviewer using "
                "`(ref: S#)` markers. Tavily must be configured to populate this."
            )
            if sources:
                table_rows = [
                    {
                        "ID": s.get("id", ""),
                        "Category": s.get("category", ""),
                        "Title": s.get("title", ""),
                        "URL": s.get("url", ""),
                        "Snippet": s.get("snippet", ""),
                    }
                    for s in sources
                ]
                st.dataframe(table_rows, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No external sources were used. Set a Tavily API key in "
                    "the sidebar to enable web-grounded review."
                )

        with tab5:
            if report_path and os.path.exists(report_path):
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.markdown(content, unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️ Download Full QA Report (Markdown)",
                        data=content,
                        file_name=os.path.basename(report_path),
                        mime="text/markdown",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Failed to load saved report: {e}")
            else:
                st.warning("Report file was not created. Check logs for errors.")
