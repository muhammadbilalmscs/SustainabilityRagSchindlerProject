"""
rag_pipeline.py - Phases 5 and 6: connect all the pieces and ask Gemini.

Responsibility
    prepare_document():  PDF bytes -> pages, chunks (text + tables), vectors, FAISS index
    answer_question():   question -> retrieve -> build context -> Gemini -> checked answer

The pipeline (one question):

    question
      -> embed_query()            question vector              (embeddings.py)
      -> search()                 top 5 chunks + scores        (vector_store.py)
      -> build_context()          numbered sources with pages  (this file)
      -> ask_gemini()             answer + figures as JSON     (this file)
      -> verify_figures()         drop numbers not in sources  (this file)
      -> result dict              shown by app.py

Why Gemini, and why the google-genai package?
    google-genai is Google's current official SDK ("from google import genai").
    The older google-generativeai package is deprecated, so tutorials that use
    genai.GenerativeModel(...) are out of date.

Hallucination prevention (three layers)
    1. Retrieval: Gemini only sees the 5 retrieved sources, not its memory of Schindler.
    2. Instructions: the system prompt forbids outside knowledge, demands page
       citations, and requires saying "not in the document" when that is true.
    3. Verification: every number Gemini returns for a chart is searched for in
       the source text of the page it cites. Numbers that are not found are dropped
       and reported, never charted.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from chunking import chunk_pages
from embeddings import embed_chunks, embed_query
from pdf_processor import extract_pages
from table_extractor import extract_tables, number_heavy_pages, tables_to_chunks
from vector_store import build_index, search


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
TOP_K = 5          # best chunks overall
EXTRA_TABLES = 2   # plus the best table chunks, if not already among them (see retrieve())


# ---------------------------------------------------------------------------
# Phase 5: Gemini connection
# ---------------------------------------------------------------------------

def get_api_key():
    """
    Read GEMINI_API_KEY from the environment, loading this project's .env file first.

    Locally:          put the key in .env (copy .env.example). .env is git-ignored.
    Streamlit Cloud:  app.py reads st.secrets first and passes the key itself.
    Output: the key (str) or None if it is not set.
    """
    project_env_file = Path(__file__).parent / ".env"
    load_dotenv(project_env_file)
    return os.getenv("GEMINI_API_KEY")


def get_model_name():
    """The Gemini model to use; override with GEMINI_MODEL in .env or secrets."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def create_client(api_key):
    """Create the Gemini client. Input: API key (str). Output: genai.Client."""
    return genai.Client(api_key=api_key)


# The exact JSON shape we want back from Gemini. Pydantic classes describe the
# fields and their types; Gemini is told to fill this shape ("structured output"),
# so we get data we can put straight into a DataFrame instead of free text.
class Figure(BaseModel):
    label: str = Field(description="What is measured, e.g. 'Total scope 1 emissions' or 'Natural gas'")
    year: str = Field(description="Reporting year as written in the source, e.g. '2025'. Empty if no year is given.")
    value: float = Field(description="The number exactly as reported, without thousands separators")
    unit: str = Field(description="Unit exactly as reported, e.g. 'MWh', 't CO2e', '%'")
    page: int = Field(description="Page number of the source the value comes from")


class RagAnswer(BaseModel):
    answer: str = Field(description="Markdown answer with page citations like (p. 18)")
    has_sufficient_information: bool = Field(description="False if the sources do not answer the question")
    figures: list[Figure] = Field(description="Numbers relevant to the question, copied from the sources. Empty if none.")


SYSTEM_INSTRUCTION = """You answer questions about one document, the Schindler Nonfinancial Report 2025, using ONLY the numbered sources given to you.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge, and never guess.
2. Cite the page for every fact, in the form (p. 18). Use the page numbers shown in the source headers.
3. If the sources do not contain the answer, say so clearly and set has_sufficient_information to false. A partial answer is fine if you say which part is missing.
4. Keep reported facts and interpretation apart. Put any interpretation, trend, or conclusion of your own in a final paragraph that starts with "**Interpretation:**".
5. Numbers: copy values exactly as reported and always state their unit and reporting year. Never convert units, never add or average values yourself, and never compare values that have different units.
6. figures: when the question is about numbers, list each relevant value (label, year, value, unit, page). Only include values that appear literally in the sources. Leave the list empty for non-numerical questions.
7. Tables appear as Markdown rows. In key-figure tables, columns are reporting years, and the unit is in the top-left header cell (e.g. "MWh", "t CO₂e", "t"). Numbers use spaces as thousands separators: "65 671" means 65671.
Be concise and well structured."""


def ask_gemini(client, question, context, model_name=None):
    """
    Send the question and the retrieved context to Gemini.

    Input:  client (genai.Client), question (str), context (str from build_context)
    Output: RagAnswer object (answer text, sufficiency flag, list of Figure)
    """
    prompt = f"SOURCES:\n{context}\n\nQUESTION: {question}"

    response = client.models.generate_content(
        model=model_name or get_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",  # ask for JSON, not prose
            response_schema=RagAnswer,              # ... in exactly this shape
        ),
    )

    # The SDK converts the JSON into our RagAnswer class for us.
    if response.parsed is not None:
        return response.parsed
    # Fallback: parse the raw text ourselves (e.g. if the SDK could not).
    return RagAnswer(**json.loads(response.text))


# ---------------------------------------------------------------------------
# Phase 6: the full pipeline
# ---------------------------------------------------------------------------

def prepare_document(pdf_bytes, embed_model):
    """
    Process a PDF once, so questions can be answered quickly afterwards.

    Input:  pdf_bytes, embed_model (from embeddings.load_model)
    Output: dict with
            pages   - from pdf_processor
            tables  - from table_extractor (all tables, with DataFrames)
            chunks  - text chunks followed by table chunks; chunk_id == FAISS row
            index   - FAISS index over all chunks
    """
    pages = extract_pages(pdf_bytes)
    text_chunks = chunk_pages(pages, tokenizer=embed_model.tokenizer)
    for chunk in text_chunks:
        chunk["source"] = "text"

    # Table detection is slow (~3 s per page), so only search number-heavy pages.
    tables = extract_tables(pdf_bytes, page_numbers=number_heavy_pages(pages))
    # Table chunk ids continue after the text chunk ids, so every id == its FAISS row.
    table_chunks = tables_to_chunks(tables, first_chunk_id=len(text_chunks),
                                    tokenizer=embed_model.tokenizer)

    chunks = text_chunks + table_chunks
    vectors = embed_chunks(embed_model, chunks)
    index = build_index(vectors)

    return {"pages": pages, "tables": tables, "chunks": chunks, "index": index}


def retrieve(question, embed_model, document, top_k=TOP_K, extra_tables=EXTRA_TABLES):
    """
    Question -> question vector -> FAISS search -> the most relevant chunks.

    Returns the top_k chunks overall PLUS the best `extra_tables` table chunks
    if they are not already included (so up to top_k + extra_tables results).

    Why the extra tables? A table chunk is mostly numbers, which carry little
    meaning for an embedding model, so tables score lower than prose even when
    they hold the exact answer. Testing showed "What were the scope 1 emissions?"
    ranked five prose chunks above the scope 1 table. Always adding the best two
    tables gives Gemini the figures, and costs only two extra sources.
    """
    query_vector = embed_query(embed_model, question)
    # Rank ALL chunks (fine for ~100 chunks), then pick from the ranked list.
    ranked = search(document["index"], document["chunks"], query_vector, top_k=len(document["chunks"]))

    results = ranked[:top_k]
    tables_added = 0
    for result in ranked[top_k:]:
        if tables_added == extra_tables:
            break
        if result.get("source") == "table":
            results.append(result)
            tables_added += 1
    return results


def build_context(results):
    """
    Turn the retrieved chunks into one text block for the LLM.

    Each source gets a header with a number, its page and its type, so Gemini
    can cite pages and we can show exactly what it was given:

        [Source 1 | page 18 | text]
        In 2025, Schindler achieved its goal of sourcing ...

    Input:  list of result dicts from search()
    Output: str
    """
    parts = []
    for number, result in enumerate(results, start=1):
        header = f"[Source {number} | page {result['page_number']} | {result.get('source', 'text')}]"
        parts.append(header + "\n" + result["text"])
    return "\n\n".join(parts)


def normalize_numbers(text):
    """
    Remove thousands separators so numbers can be compared:
    "65 671" -> "65671", "1,440" -> "1440", "–31" -> "-31".
    """
    text = text.replace("–", "-").replace(" ", " ")
    # A space or comma between digits, followed by exactly 3 digits, is a thousands separator.
    return re.sub(r"(?<=\d)[ ,](?=\d{3}(?!\d))", "", text)


def number_variants(value):
    """
    The ways a value may be printed: 65671.0 -> ["65671"]; 59.5 -> ["59.5"];
    2.0 -> ["2", "2.0"]. Negative values are also searched without the sign
    ("–31%" is often written as "reduction of 31%").
    """
    variants = []
    magnitude = abs(value)
    if magnitude == int(magnitude):
        variants.append(str(int(magnitude)))
    variants.append(f"{magnitude:g}")
    variants.append(str(magnitude))
    unique = []
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return unique


def verify_figures(figures, results):
    """
    Keep only the figures whose value really appears on the cited page in the sources.

    Input:  figures (list of Figure), results (the retrieved chunks Gemini saw)
    Output: (verified, rejected) - two lists of plain dicts

    Example: Gemini says "Natural gas, 2025, 65671, MWh, page 37".
             We look for 65671 in the page-37 sources -> not found -> rejected.
             (It is on page 36.) A wrong citation is caught just like an invented number.
    """
    text_by_page = {}
    for result in results:
        page_text = text_by_page.get(result["page_number"], "")
        text_by_page[result["page_number"]] = page_text + "\n" + normalize_numbers(result["text"])

    verified = []
    rejected = []
    for figure in figures:
        figure_dict = figure.model_dump() if hasattr(figure, "model_dump") else dict(figure)
        page_text = text_by_page.get(figure_dict["page"], "")
        found = False
        for variant in number_variants(figure_dict["value"]):
            # The number must stand alone: 671 must not match inside 65671.
            pattern = r"(?<![\d.])" + re.escape(variant) + r"(?![\d])"
            if re.search(pattern, page_text):
                found = True
                break
        if found:
            verified.append(figure_dict)
        else:
            rejected.append(figure_dict)
    return verified, rejected


def answer_question(question, embed_model, document, client, top_k=TOP_K, model_name=None):
    """
    The complete RAG pipeline for one question.

    Input:  question, embed_model, document (from prepare_document), Gemini client
    Output: dict with
            answer          - Markdown text with page citations
            sufficient      - did the sources contain the answer?
            figures         - verified numbers (list of dicts) for tables/charts
            rejected        - numbers Gemini gave that we could NOT find in the sources
            pages           - cited source pages, sorted
            results         - the retrieved chunks with scores (for the debug view)
            context         - the exact text sent to Gemini (for the debug view)
    """
    results = retrieve(question, embed_model, document, top_k)
    context = build_context(results)
    reply = ask_gemini(client, question, context, model_name)
    verified, rejected = verify_figures(reply.figures, results)

    pages = sorted({result["page_number"] for result in results})
    return {
        "answer": reply.answer,
        "sufficient": reply.has_sufficient_information,
        "figures": verified,
        "rejected": rejected,
        "pages": pages,
        "results": results,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Quick manual test:
#   python rag_pipeline.py data/schindler_report.pdf "What were the scope 1 emissions?"
# Without an API key it still shows the retrieval and the context it would send.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from embeddings import load_model

    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print('Usage: python rag_pipeline.py path/to/file.pdf ["your question"]')
        sys.exit(1)
    question = sys.argv[2] if len(sys.argv) > 2 else "What were Schindler's scope 1 emissions in 2025 and 2024?"

    embed_model = load_model()
    with open(sys.argv[1], "rb") as pdf_file:
        document = prepare_document(pdf_file.read(), embed_model)
    text_count = sum(1 for c in document["chunks"] if c["source"] == "text")
    print(f"Prepared: {len(document['pages'])} pages, {text_count} text chunks, "
          f"{len(document['chunks']) - text_count} table chunks, {len(document['tables'])} tables")

    results = retrieve(question, embed_model, document)
    print(f"\nQuestion: {question}\nRetrieved:")
    for result in results:
        print(f"  #{result['rank']} score {result['score']:.3f}  page {result['page_number']:2d}  {result['source']}")

    api_key = get_api_key()
    if not api_key:
        print("\nNo GEMINI_API_KEY found. Copy .env.example to .env, add your key, and run again.")
        print("Context that would be sent (first 800 characters):\n")
        print(build_context(results)[:800])
        sys.exit(0)

    output = answer_question(question, embed_model, document, create_client(api_key))
    print(f"\n--- Answer ({get_model_name()}) ---\n{output['answer']}")
    print(f"\nSufficient information: {output['sufficient']}")
    print(f"Verified figures: {len(output['figures'])}   Rejected: {len(output['rejected'])}")
    for figure in output["figures"]:
        print(f"  {figure['label']} | {figure['year']} | {figure['value']:g} {figure['unit']} | p. {figure['page']}")
