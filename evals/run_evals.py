"""
Lightweight evaluation harness (Gap #12 scaffold).

Runs each requirement in ``golden_dataset.json`` through the QA Intelligence
Suite graph and scores the output along three axes:

* **Coverage**: every ``must_have_keyword`` appears at least once in the
  generated test suite (case-insensitive substring match).
* **Categories**: every ``must_have_categories`` value appears in the suite.
* **Volume**: at least ``min_test_cases`` distinct ``TC-NNN`` IDs exist.

This is heuristic-only — a *real* eval would add an LLM-as-judge for
specificity/Gherkin-correctness. The harness is structured so that judge
can be plugged in as a fourth scorer without rewriting the runner.

Run from repo root:

    python -m evals.run_evals

Requires the same environment as the app (GROQ_API_KEY or GEMINI_API_KEY,
optional TAVILY_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.langgraphAgenticAI.graph.graph_builder import GraphBuilder  # noqa: E402
from src.langgraphAgenticAI.observability.setup import setup_observability  # noqa: E402

DATASET = REPO_ROOT / "evals" / "golden_dataset.json"
TC_ID_RE = re.compile(r"\bTC-\d+\b")


def _build_default_llm():
    """Pick whichever provider is configured; fail loudly if none."""
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.environ.get("EVAL_GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            max_retries=3,
            timeout=60,
        )
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.environ.get("EVAL_GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0,
            max_retries=3,
            timeout=60,
        )
    raise RuntimeError(
        "No LLM credentials found. Set GROQ_API_KEY or GEMINI_API_KEY."
    )


def _score_one(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    suite = (result.get("test_cases") or "").lower()
    expected = case["expected_coverage"]

    keyword_hits = {
        kw: (kw.lower() in suite)
        for kw in expected.get("must_have_keywords", [])
    }
    category_hits = {
        cat: (cat.lower() in suite)
        for cat in expected.get("must_have_categories", [])
    }
    tc_ids = set(TC_ID_RE.findall(result.get("test_cases") or ""))
    volume_ok = len(tc_ids) >= int(expected.get("min_test_cases", 0))

    score = (
        sum(keyword_hits.values())
        + sum(category_hits.values())
        + (1 if volume_ok else 0)
    )
    max_score = (
        len(keyword_hits) + len(category_hits) + 1
    )

    return {
        "case_id": case["id"],
        "score": f"{score}/{max_score}",
        "score_pct": round(100.0 * score / max_score, 1) if max_score else 0.0,
        "keyword_hits": keyword_hits,
        "category_hits": category_hits,
        "tc_count": len(tc_ids),
        "volume_ok": volume_ok,
    }


def main() -> int:
    setup_observability()
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    llm = _build_default_llm()
    graph = GraphBuilder(model=llm).setup_graph(GraphBuilder.USECASE)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[eval] running case={case['id']} ...")
        result = graph.invoke(
            {"requirement": case["requirement"]},
            config={"recursion_limit": 12},
        )
        rows.append(_score_one(case, result))

    print("\n=== Eval Summary ===")
    print(json.dumps(rows, indent=2))
    avg = sum(r["score_pct"] for r in rows) / max(len(rows), 1)
    print(f"\nAverage score: {avg:.1f}%")
    # Treat <60% average as a regression for CI gating.
    return 0 if avg >= 60.0 else 1


if __name__ == "__main__":
    sys.exit(main())
