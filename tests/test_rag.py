"""
Tests for the RAG assistant.   Run:  pytest          (all tests)
                                     pytest -m "not slow"   (fast tests only, ~3 s)

Fast tests use tiny hand-made inputs, so they need no PDF, no model and no API key.
Slow tests use the real report in data/ and the real embedding model, but never
call Gemini: ask_gemini is replaced by a fake, so the tests are free and repeatable.
"""

from pathlib import Path

import numpy as np
import pytest

import rag_pipeline
from chunking import split_text
from rag_pipeline import Figure, RagAnswer, normalize_numbers, verify_figures
from table_extractor import parse_number
from vector_store import build_index, search
from visualizations import figures_to_dataframe, group_by_unit, make_chart

PDF_PATH = Path(__file__).parent.parent / "data" / "schindler_report.pdf"


# ----------------------------- fast unit tests -----------------------------

def test_split_text_overlaps_and_respects_size():
    text = " ".join(f"w{i}" for i in range(10))
    # Without a tokenizer each word counts as 1.33 tokens: size 5 tokens = 3 words, overlap 1.4 = 1 word.
    pieces = split_text(text, chunk_size_tokens=5, overlap_tokens=1.4)
    assert pieces[0] == "w0 w1 w2"
    assert pieces[1].startswith("w2")          # the last word is repeated (overlap)
    assert pieces[-1].endswith("w9")           # nothing is lost at the end


def test_parse_number_refuses_ambiguous_values():
    assert parse_number("65 671") == 65671
    assert parse_number("–31%") == -31
    assert parse_number("2.7") == 2.7
    assert parse_number("7 124 197 12 370 288") is None   # two merged numbers
    assert parse_number("Natural gas") is None


def test_normalize_numbers_removes_thousands_separators():
    assert normalize_numbers("Natural gas 65 671 MWh, fuel 1,440") == "Natural gas 65671 MWh, fuel 1440"
    assert normalize_numbers("in 2025 2024") == "in 2025 2024"   # two years stay two numbers


def test_verify_figures_rejects_invented_numbers_and_wrong_pages():
    results = [{"page_number": 36, "text": "| Natural gas | 65 671 | 69 336 |"}]
    figures = [
        Figure(label="Natural gas", year="2025", value=65671, unit="MWh", page=36),  # correct
        Figure(label="Natural gas", year="2023", value=70000, unit="MWh", page=36),  # invented
        Figure(label="Natural gas", year="2024", value=69336, unit="MWh", page=37),  # wrong page
        Figure(label="Part of a number", year="", value=671, unit="MWh", page=36),   # only a fragment
    ]
    verified, rejected = verify_figures(figures, results)
    assert [f["value"] for f in verified] == [65671]
    assert len(rejected) == 3


def test_faiss_search_returns_closest_vector_first():
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], dtype="float32")
    chunks = [{"chunk_id": i, "page_number": i + 1, "text": f"chunk {i}"} for i in range(3)]
    index = build_index(vectors)
    results = search(index, chunks, np.array([[0, 1, 0]], dtype="float32"), top_k=2)
    assert [r["chunk_id"] for r in results] == [1, 2]
    assert results[0]["score"] == pytest.approx(1.0)


def test_charts_never_mix_units_and_need_two_values():
    df = figures_to_dataframe([
        {"label": "Scope 1", "year": "2025", "value": 125756, "unit": "t CO₂e", "page": 37},
        {"label": "Scope 1", "year": "2024", "value": 131225, "unit": "t CO2e", "page": 37},
        {"label": "Natural gas", "year": "2025", "value": 65671, "unit": "MWh", "page": 36},
    ])
    groups = group_by_unit(df)
    assert len(groups) == 2                      # t CO₂e and MWh are kept apart
    charts = {unit: make_chart(group, unit) for unit, group in groups.items()}
    assert charts["MWh"] is None                 # one value -> KPI, not a chart
    assert sum(chart is not None for chart in charts.values()) == 1


# ------------------------ slow tests with the real PDF ------------------------

needs_pdf = pytest.mark.skipif(not PDF_PATH.exists(), reason="put the report at data/schindler_report.pdf")


@pytest.fixture(scope="module")
def prepared():
    from embeddings import load_model
    model = load_model()
    document = rag_pipeline.prepare_document(PDF_PATH.read_bytes(), model)
    return model, document


@pytest.mark.slow
@needs_pdf
def test_document_preparation(prepared):
    model, document = prepared
    assert len(document["pages"]) == 53
    assert document["index"].ntotal == len(document["chunks"])
    # chunk_id must equal the FAISS row, or search results would point at the wrong text
    assert [c["chunk_id"] for c in document["chunks"]] == list(range(len(document["chunks"])))
    assert max(c["token_count"] for c in document["chunks"] if c["source"] == "text") <= 500


# A small retrieval evaluation: for each question, the page that holds the answer
# must be among the retrieved pages. Re-run this after changing chunk size,
# model or top_k to see whether retrieval got better or worse.
RETRIEVAL_CASES = [
    ("What were the scope 1 emissions in 2025?", 37),
    ("How much natural gas did Schindler consume?", 36),
    ("Did Schindler reach 100% renewable electricity?", 18),
    ("How does Schindler address child labor risks?", 26),
    ("How much hazardous waste was sent to landfill?", 38),
    ("Who performed the limited assurance engagement?", 49),
]


@pytest.mark.slow
@needs_pdf
@pytest.mark.parametrize("question, expected_page", RETRIEVAL_CASES)
def test_retrieval_finds_the_right_page(prepared, question, expected_page):
    model, document = prepared
    pages = [r["page_number"] for r in rag_pipeline.retrieve(question, model, document)]
    assert expected_page in pages


def fake_ask_gemini(client, question, context, model_name=None):
    """Stands in for Gemini: two real figures from page 37 and one invented one."""
    return RagAnswer(
        answer="Scope 1 emissions were 125,756 t CO₂e in 2025 (p. 37).",
        has_sufficient_information=True,
        figures=[
            Figure(label="Total scope 1", year="2025", value=125756, unit="t CO₂e", page=37),
            Figure(label="Total scope 1", year="2024", value=131225, unit="t CO₂e", page=37),
            Figure(label="Total scope 1", year="2023", value=128000, unit="t CO₂e", page=37),
        ],
    )


@pytest.mark.slow
@needs_pdf
def test_app_end_to_end(monkeypatch):
    """Upload the PDF in the real app, ask a question, and check the answer, chart and debug view."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(rag_pipeline, "ask_gemini", fake_ask_gemini)
    at = AppTest.from_file(str(PDF_PATH.parent.parent / "app.py"), default_timeout=300)
    at.secrets["GEMINI_API_KEY"] = "test-key-not-used"
    at.run()
    assert not at.exception

    at.file_uploader[0].upload("schindler_report.pdf", PDF_PATH.read_bytes())
    at.run()
    assert not at.exception
    assert [m.value for m in at.sidebar.metric][:1] == ["53"]

    at.chat_input[0].set_value("What were the scope 1 emissions?")
    at.run()
    assert not at.exception
    assert any("125,756" in text for text in at.markdown.values)  # the answer is shown
    assert len(at.get("plotly_chart")) == 1                   # 2 verified values -> 1 chart
    assert any("not found in the sources" in c for c in at.caption.values)   # 2023 value was rejected
