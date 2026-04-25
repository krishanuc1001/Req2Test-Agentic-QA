from configparser import ConfigParser


class UIConfigReader:
    """Reads UI configuration values for the QA Intelligence Suite app."""

    def __init__(self, config_file="src/langgraphAgenticAI/ui/ui_config.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)

    def _get_csv(self, key: str) -> list:
        raw = self.config["DEFAULT"].get(key, "")
        return [opt.strip() for opt in raw.split(",") if opt.strip()]

    def get_lmm_options(self) -> list:
        return self._get_csv("LMM_OPTIONS")

    def get_groq_model_options(self) -> list:
        return self._get_csv("GROQ_MODEL_OPTIONS")

    def get_gemini_model_options(self) -> list:
        return self._get_csv("GEMINI_MODEL_OPTIONS")

    def get_page_title(self) -> str:
        return self.config["DEFAULT"].get("PAGE_TITLE", "QA Intelligence Suite")

    def get_usecase(self) -> str:
        return self.config["DEFAULT"].get("USECASE", "QA Intelligence Suite")
