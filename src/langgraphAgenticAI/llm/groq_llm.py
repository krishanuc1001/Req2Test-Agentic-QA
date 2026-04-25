import os
import streamlit as st
from langchain_groq import ChatGroq


def _read_secret(key: str) -> str:
    """
    Safely read a secret from `st.secrets` without raising when no
    secrets.toml is configured (e.g. local dev). Falls back to env var.
    """
    try:
        value = st.secrets.get(key, "")
        if value:
            return str(value)
    except Exception:
        # Streamlit raises StreamlitSecretNotFoundError when secrets.toml
        # does not exist; treat as "no secret available".
        pass
    return os.environ.get(key, "")


class GroqLLM:
    """
    Builds a ChatGroq LLM client.

    API key resolution order:
      1. Sidebar text input (user_controls_input["GROQ_API_KEY"])
      2. Streamlit deployment secret (st.secrets["GROQ_API_KEY"])
      3. Environment variable GROQ_API_KEY
    """

    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        groq_api_key = (
            self.user_controls_input.get("GROQ_API_KEY")
            or _read_secret("GROQ_API_KEY")
        )
        selected_groq_model = self.user_controls_input.get("selected_groq_model")

        if not groq_api_key:
            st.error(
                "GROQ API Key is required. Provide it in the sidebar, "
                "or configure `GROQ_API_KEY` as a Streamlit secret."
            )
            return None

        if not selected_groq_model:
            st.error("Please select a Groq model in the sidebar.")
            return None

        try:
            # Gap #4: explicit retry + timeout guardrails so transient 429/5xx
            # do not blow up the whole agent run.
            return ChatGroq(
                model=selected_groq_model,
                api_key=groq_api_key,
                temperature=0,
                max_retries=3,
                timeout=60,
            )
        except Exception as e:
            raise ValueError(f"Error configuring Groq LLM: {e}")