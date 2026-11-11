"""
app.py - Phase 8: the Streamlit frontend.

Run it with:   streamlit run app.py

How Streamlit works (the one idea to understand)
    Streamlit runs this whole file from top to bottom every time the user does
    something (uploads a file, sends a question). Two tools stop slow work from
    repeating on every run:
      - @st.cache_resource remembers the result of a function (the embedding model,
        the processed PDF) and returns it instantly on the next run.
      - st.session_state is a dictionary that survives between runs; we keep the
        chat history in it.

Data flow
    upload -> process_pdf()  (cached: pages, chunks, tables, FAISS index)
    question -> rag_pipeline.answer_question() -> answer + figures
    figures -> visualizations -> KPIs, table, Plotly charts
"""

import hashlib

import streamlit as st
from google.genai import errors as genai_errors

import rag_pipeline
from embeddings import get_model_info, load_model
from visualizations import figures_to_dataframe, group_by_unit, make_chart


st.set_page_config(page_title="Sustainability report assistant", page_icon=":material/eco:", layout="wide")

EXAMPLE_QUESTIONS = [
    "What were the scope 1 and 2 emissions in 2025, 2024 and 2020?",
    "How much energy did the company consume?",
    "What share of electricity came from renewable sources?",
    "How much hazardous and nonhazardous waste was produced?",
    "What is Schindler's net-zero target?",
]


# ---------------------------------------------------------------------------
# Cached resources: created once, reused across reruns and users
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model (first time only)...")
def get_embedding_model():
    return load_model()


@st.cache_resource(max_entries=2, show_spinner=False)
def process_pdf(pdf_bytes, _embed_model):
    """
    Process a PDF once per file. Streamlit uses pdf_bytes as the cache key, so
    the same file is not processed twice. The leading underscore in
    _embed_model tells Streamlit not to try to hash the model.

    Nothing is written to disk: on Streamlit Community Cloud the disk is not
    permanent, so we rebuild the index from the upload instead of saving it.
    """
    return rag_pipeline.prepare_document(pdf_bytes, _embed_model)


def get_api_key():
    """Streamlit Cloud: st.secrets. Local: .env (via rag_pipeline.get_api_key)."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets.toml file at all - normal when running locally
        pass
    return rag_pipeline.get_api_key()


@st.cache_resource
def get_gemini_client(api_key):
    return rag_pipeline.create_client(api_key)


# ---------------------------------------------------------------------------
# Showing one answer (used for new answers and for the chat history)
# ---------------------------------------------------------------------------

def format_value(value):
    """65671.0 -> '65,671'   2.7 -> '2.7'"""
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def show_figures(result, message_number):
    """Show verified figures as KPIs (single values) or charts (2+ values), plus a table."""
    df = figures_to_dataframe(result["figures"])
    if df.empty:
        return

    st.markdown("**Figures from the report**")
    for unit, group in group_by_unit(df).items():
        chart = make_chart(group, unit)
        if chart is None:
            row = group.iloc[0]
            st.metric(
                label=f"{row['Label']} ({row['Year'] or 'year not stated'})",
                value=f"{format_value(row['Value'])} {unit}",
                help=f"Source: page {row['Page']}",
                border=True,
                width="content",
            )
        else:
            st.plotly_chart(chart, key=f"chart_{message_number}_{unit}",
                            alt=f"Bar chart of {unit} values from the report")

    st.dataframe(
        df, hide_index=True, key=f"figures_{message_number}",
        alt="Table of figures used in the answer",
        column_config={"Value": st.column_config.NumberColumn(format="localized")},
    )


def show_debug(result, message_number):
    """The 'how did the RAG get this answer?' section."""
    with st.expander("Retrieval details", icon=":material/manage_search:", key=f"debug_{message_number}"):
        st.markdown(f"**Retrieved chunks** (top {rag_pipeline.TOP_K} + up to "
                    f"{rag_pipeline.EXTRA_TABLES} best tables). Higher score = closer in meaning.")
        rows = []
        for item in result["results"]:
            rows.append({
                "Rank": item["rank"],
                "Score": item["score"],
                "Page": item["page_number"],
                "Type": item.get("source", "text"),
                "Text": item["text"][:300],
            })
        st.dataframe(rows, hide_index=True, key=f"retrieved_{message_number}",
                     alt="Retrieved chunks with similarity scores")

        if result["rejected"]:
            st.markdown("**Figures removed by verification** (not found on the cited page):")
            st.dataframe(figures_to_dataframe(result["rejected"]), hide_index=True,
                         key=f"rejected_{message_number}", alt="Figures that failed verification")

        st.markdown("**Exact context sent to Gemini**")
        st.code(result["context"], language=None, wrap_lines=True, height=300)


def show_answer(result, message_number):
    st.markdown(result["answer"])
    if not result["sufficient"]:
        st.warning("The retrieved parts of the report do not fully answer this question.",
                   icon=":material/info:")
    if result["rejected"]:
        st.caption(f":material/shield: {len(result['rejected'])} number(s) suggested by the model "
                   "were not found in the sources and were left out.")
    show_figures(result, message_number)
    st.caption("Pages searched: " + ", ".join(str(page) for page in result["pages"]))
    show_debug(result, message_number)


# ---------------------------------------------------------------------------
# Sidebar: upload and processing status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF report", type="pdf")

    document = None
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        file_id = hashlib.md5(pdf_bytes).hexdigest()

        # A different file means a new conversation.
        if st.session_state.get("file_id") != file_id:
            st.session_state.file_id = file_id
            st.session_state.messages = []

        embed_model = get_embedding_model()
        with st.status("Processing the PDF...", expanded=False) as status:
            try:
                document = process_pdf(pdf_bytes, embed_model)
                status.update(label="Document ready", state="complete")
            except ValueError as error:
                status.update(label="Could not process this PDF", state="error")
                st.error(str(error))

    if document is not None:
        text_chunks = sum(1 for chunk in document["chunks"] if chunk["source"] == "text")
        empty_pages = sum(1 for page in document["pages"] if page["is_empty"])
        with st.container(horizontal=True):
            st.metric("Pages", len(document["pages"]), border=True)
            st.metric("Chunks", text_chunks, border=True)
            st.metric("Tables", len(document["tables"]), border=True)
        if empty_pages:
            st.caption(f"{empty_pages} page(s) had no text (scanned or blank) and were skipped.")

        with st.expander("Embedding model", icon=":material/memory:"):
            info = get_model_info(get_embedding_model())
            st.markdown(f"- **Model:** `{info['name']}`\n"
                        f"- **Vector size:** {info['dimensions']} numbers\n"
                        f"- **Max input:** {info['max_tokens']} tokens\n"
                        f"- **Index:** FAISS IndexFlatIP, {document['index'].ntotal} vectors\n"
                        f"- **LLM:** `{rag_pipeline.get_model_name()}`")

    st.caption("The PDF is processed in memory only and is not stored.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Sustainability report assistant")
st.caption("Ask questions about an uploaded report. Answers cite page numbers and use only the document.")

if document is None:
    st.info("Upload a PDF in the sidebar to start, for example the Schindler Nonfinancial Report 2025.",
            icon=":material/upload_file:")
    st.stop()

api_key = get_api_key()
if not api_key:
    st.error("Add GEMINI_API_KEY to a .env file (local) or to the app's Secrets (Streamlit Cloud).",
             title="Gemini API key missing")
    st.stop()

with st.expander("Browse extracted tables", icon=":material/table:"):
    labels = []
    for table in document["tables"]:
        labels.append(f"Page {table['page_number']} - table {table['table_on_page']}: {table['heading'][:60]}")
    if labels:
        choice = st.selectbox("Table", range(len(labels)), format_func=lambda i: labels[i])
        st.dataframe(document["tables"][choice]["dataframe"], hide_index=True,
                     alt="Table extracted from the PDF")
    else:
        st.write("No tables were found in this document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far (Streamlit reruns the script on every question).
for number, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message["result"], number)

# Example questions for an empty chat.
clicked_example = None
if not st.session_state.messages:
    st.markdown("**Try a question:**")
    with st.container(horizontal=True):
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example_{example}"):
                clicked_example = example

typed_question = st.chat_input("Ask about the report", submit_mode="disable")
question = typed_question or clicked_example

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the report and asking Gemini..."):
                result = rag_pipeline.answer_question(
                    question, get_embedding_model(), document, get_gemini_client(api_key))
        except genai_errors.APIError as error:
            if error.code in rag_pipeline.RETRYABLE_STATUS_CODES:
                st.warning("Gemini is busy right now and did not answer after several tries. "
                           "Please ask again in a minute.", title="Gemini is overloaded")
            else:
                st.error(f"Gemini returned an error ({error.code}): {error.message}", title="LLM request failed")
            st.session_state.messages.pop()  # forget the unanswered question
            st.stop()

        number = len(st.session_state.messages)
        show_answer(result, number)

    st.session_state.messages.append({"role": "assistant", "result": result})
    if clicked_example:
        st.rerun()  # hide the example buttons now that the chat has started
