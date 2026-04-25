"""Tests for the citation sanitizer (Gap #5 defense-in-depth)."""
from src.langgraphAgenticAI.nodes.test_reviewer_node import _sanitize_citations


def test_keeps_valid_citation():
    text = "High-value smoke check (ref: S1)."
    out, dropped = _sanitize_citations(text, valid_ids={"S1", "S2"})
    assert out == "High-value smoke check (ref: S1)."
    assert dropped == set()


def test_strips_unknown_id_keeps_valid():
    text = "Important (ref: S1, S99)."
    out, dropped = _sanitize_citations(text, valid_ids={"S1"})
    assert out == "Important (ref: S1)."
    assert dropped == {"S99"}


def test_removes_block_when_all_unknown():
    text = "Spurious claim (ref: S77, S88)."
    out, dropped = _sanitize_citations(text, valid_ids={"S1"})
    assert "(ref:" not in out
    assert dropped == {"S77", "S88"}


def test_no_citations_unchanged():
    text = "Plain prose with no citations at all."
    out, dropped = _sanitize_citations(text, valid_ids={"S1"})
    assert out == text
    assert dropped == set()


def test_multiple_blocks_in_one_paragraph():
    text = "First (ref: S1) and second (ref: S99) and third (ref: S2)."
    out, dropped = _sanitize_citations(text, valid_ids={"S1", "S2"})
    assert "S99" not in out
    assert "S1" in out and "S2" in out
    assert dropped == {"S99"}


def test_empty_valid_set_drops_all():
    text = "Claim (ref: S1, S2)."
    out, dropped = _sanitize_citations(text, valid_ids=set())
    assert "(ref:" not in out
    assert dropped == {"S1", "S2"}
