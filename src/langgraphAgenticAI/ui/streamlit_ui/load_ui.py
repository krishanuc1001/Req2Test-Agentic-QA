import os
import streamlit as st

from src.langgraphAgenticAI.ui.ui_config_reader import UIConfigReader


class LoadStreamlitUI:
    """
    Renders the sidebar controls for the QA Intelligence Suite:
        - LLM provider + model + API key
        - Optional Tavily API key (enriches Agent 3 with web grounding)
        - Requirement text area
        - "Run Multi-Agent QA Workflow" trigger button

    The button writes the requirement into session_state and flips
    `IsQAGenerateClicked` to True for the current Streamlit rerun.
    """

    REQUIREMENT_PLACEHOLDER = (
        "As a registered user, I want to reset my password via email "
        "so that I can regain access when I forget it.\n\n"
        "Acceptance Criteria:\n"
        "- Email must be valid and registered in the system\n"
        "- Reset link expires after 30 minutes\n"
        "- New password must meet complexity policy\n"
        "- User is notified via email on successful reset"
    )

    def __init__(self):
        self.config = UIConfigReader()
        self.user_controls = {}

    def load_streamlit_ui(self) -> dict:
        st.set_page_config(page_title=self.config.get_page_title(), layout="wide")
        st.header(" 🤖 " + self.config.get_page_title())

        # Reset run-trigger on every rerun; button below flips it to True.
        st.session_state.IsQAGenerateClicked = False
        if "qa_requirement_text" not in st.session_state:
            st.session_state.qa_requirement_text = ""

        # Use case is fixed for this app
        self.user_controls["selected_usecase"] = self.config.get_usecase()

        with st.sidebar:
            llm_options = self.config.get_lmm_options()

            # ---- LLM provider selection ----
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            if self.user_controls["selected_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox(
                    "Select Model", model_options
                )
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = (
                    st.text_input("Groq API Key", type="password")
                )
                if not self.user_controls["GROQ_API_KEY"]:
                    st.info("Please enter your Groq API Key to proceed. Get it from https://groq.com/")

            elif self.user_controls["selected_llm"] == "Gemini":
                model_options = self.config.get_gemini_model_options()
                self.user_controls["selected_gemini_model"] = st.selectbox(
                    "Select Model", model_options
                )
                self.user_controls["GEMINI_API_KEY"] = st.session_state["GEMINI_API_KEY"] = (
                    st.text_input("Gemini API Key", type="password")
                )
                if not self.user_controls["GEMINI_API_KEY"]:
                    st.info("Please enter your Gemini API Key to proceed. Get it from https://ai.google.com/")

            # ---- Optional Tavily key (enriches Agent 3) ----
            tavily_key = st.text_input(
                "Tavily API Key (optional, enriches Agent 3 review)",
                type="password",
                key="qa_tavily_key_input",
            )
            # Fallback chain: sidebar input -> st.secrets -> existing env var
            if not tavily_key:
                try:
                    tavily_key = st.secrets.get("TAVILY_API_KEY", "") or ""
                except Exception:
                    tavily_key = ""
            if tavily_key:
                os.environ["TAVILY_API_KEY"] = tavily_key
                self.user_controls["TAVILY_API_KEY"] = tavily_key
                st.session_state["TAVILY_API_KEY"] = tavily_key
            else:
                # Ensure stale env var from a previous run does not bleed in.
                os.environ.pop("TAVILY_API_KEY", None)
                st.caption(
                    "ℹ️ Tavily key is optional. Without it, Agent 3 skips the "
                    "web-grounded source ledger and tool loop."
                )

            # ---- Requirement input ----
            st.subheader("🧪 QA Intelligence Suite")
            st.caption("Three agents collaborate: 🧠 Analyzer → 🧪 Generator → 🔍 Reviewer")

            requirement_text = st.text_area(
                "📝 Paste User Story / Requirement",
                height=220,
                placeholder=self.REQUIREMENT_PLACEHOLDER,
                key="qa_requirement_input",
            )

            if st.button("🚀 Run Multi-Agent QA Workflow", use_container_width=True):
                if requirement_text and requirement_text.strip():
                    st.session_state.IsQAGenerateClicked = True
                    st.session_state.qa_requirement_text = requirement_text.strip()
                else:
                    st.warning(
                        "⚠️ Please paste a user story / requirement before "
                        "running the workflow."
                    )

        return self.user_controls
