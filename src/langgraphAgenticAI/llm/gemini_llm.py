import os

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI


def _read_secret(key: str) -> str:
    """
    Safely read a secret from `st.secrets` without raising when no
    secrets.toml is configured. Falls back to env var.
    """
    try:
        value = st.secrets.get(key, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(key, "")


class GeminiLLM:
    """
    Builds a ChatGoogleGenerativeAI LLM client.

    API key resolution order:
      1. Sidebar text input (user_controls_input["GEMINI_API_KEY"])
      2. Streamlit deployment secret (st.secrets["GEMINI_API_KEY"])
      3. Environment variable GEMINI_API_KEY
    """

    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        gemini_api_key = (
            self.user_controls_input.get("GEMINI_API_KEY")
            or _read_secret("GEMINI_API_KEY")
        )
        selected_gemini_model = self.user_controls_input.get("selected_gemini_model")

        if not gemini_api_key:
            st.error(
                "GEMINI API Key is required. Provide it in the sidebar, "
                "or configure `GEMINI_API_KEY` as a Streamlit secret."
            )
            return None

        if not selected_gemini_model:
            st.error("Please select a Gemini model in the sidebar.")
            return None

        try:
            # Gap #4: explicit retry + timeout guardrails.
            return ChatGoogleGenerativeAI(
                model=selected_gemini_model,
                api_key=gemini_api_key,
                temperature=0,
                max_retries=3,
                timeout=60,
            )
        except Exception as e:
            raise ValueError(f"Error configuring Gemini LLM: {e}") from e
