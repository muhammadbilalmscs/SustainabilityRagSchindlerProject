# Sustainability Report RAG Assistant (Schindler)

A beginner-friendly **Retrieval-Augmented Generation (RAG)** app. You upload a sustainability report (PDF),
ask questions in plain English, and get answers that:

- use **only the document** and cite **page numbers**
- show the **retrieved text chunks and similarity scores**
- turn numbers into **tables, KPIs and interactive Plotly charts**, but only numbers that were
  **verified in the source text**

It was built step by step without LangChain or LlamaIndex, so every RAG step is plain, readable Python.

## How it works

```
PDF ─► pdf_processor.py ─► chunking.py ─► embeddings.py ─► vector_store.py (FAISS)
         page text          ~500-token        384-number        in-memory index
                            chunks + pages    vectors
PDF ─► table_extractor.py ─► key-figure tables as Markdown chunks (also indexed)

Question ─► embed ─► FAISS top 5 (+2 best tables) ─► context with page headers
         ─► Gemini (JSON: answer + figures) ─► verify every number ─► answer, KPIs, charts
```

| File | Responsibility |
|---|---|
| `pdf_processor.py` | Extract and clean text page by page (PyMuPDF). Removes repeated headers and invisible characters. |
| `chunking.py` | Split pages into overlapping chunks, measured in the embedding model's real tokens. |
| `embeddings.py` | Load `BAAI/bge-small-en-v1.5` and turn text into normalized 384-number vectors. |
| `vector_store.py` | Build a FAISS `IndexFlatIP` (cosine similarity) and search it. |
| `table_extractor.py` | Find tables (PyMuPDF `find_tables`) and turn numeric tables into searchable chunks. |
| `rag_pipeline.py` | Retrieval, context building, the Gemini call with structured output, and number verification. |
| `visualizations.py` | Figures to DataFrame, one chart per unit (units are never mixed). |
| `app.py` | Streamlit interface: upload, chat, charts, retrieval details. |
| `tests/test_rag.py` | Unit tests, a small retrieval evaluation, and an end-to-end app test (Gemini mocked). |

**Design note:** the project brief suggested pdfplumber for tables. The Schindler tables have no ruling
lines, so pdfplumber found only 1 table in the whole report. PyMuPDF found 18 usable ones, so we use PyMuPDF.

## Run locally

Requires Python 3.11 or newer (developed on 3.14).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env               # then put your key in .env   (macOS/Linux: cp)
streamlit run app.py
```

Get a Gemini API key at <https://aistudio.google.com/apikey>. The model defaults to `gemini-3.8-flash`;
set `GEMINI_MODEL` in `.env` to change it.

Each module can also be run on its own to see what it does, for example
`python chunking.py data/schindler_report.pdf`.

### Tests

```bash
pytest -m "not slow"     # fast unit tests, no PDF / model / API key needed
pytest                   # everything; slow tests need the report at data/schindler_report.pdf
```

## Deploy to Streamlit Community Cloud

### 1. Keep secrets and documents out of GitHub

`.gitignore` already excludes:

- `.env`, `.streamlit/secrets.toml` and other key files, so **API keys are never committed**
- `data/`, `*.pdf`, `*.docx` and `*.xlsx`, so **the report is never committed** (users upload it in the app)
- `.venv/` and caches

Before every push, check what will be committed:

```bash
git status
git diff --cached --name-only
```

If a key is ever pushed by mistake, **revoke it in Google AI Studio immediately** and create a new one.
Deleting the file afterwards is not enough, because the key stays in the Git history.

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "RAG assistant for sustainability reports"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 3. Create the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app**, then choose your repository, branch `main` and main file `app.py`.
3. Open **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste
     ```toml
     GEMINI_API_KEY = "your-real-key"
     ```
4. Click **Deploy**. The first build takes several minutes (PyTorch is large).

`app.py` reads `st.secrets["GEMINI_API_KEY"]` on Streamlit Cloud and falls back to `.env` locally.

### 4. How uploaded PDFs are handled

- The upload stays **in memory**. It is never written to disk or stored anywhere.
- Community Cloud has **no permanent local storage**, so the app does not save FAISS files. It rebuilds
  the index from the uploaded PDF (about 15–30 seconds) and caches it in memory with `st.cache_resource`
  while the app is running.
- When the app restarts or sleeps, the cache is cleared and users simply upload the PDF again.

### Troubleshooting the deployment

- **Build is slow or runs out of space:** on Linux, `pip install torch` downloads GPU (CUDA) libraries that
  are not needed here. If the build fails, install the CPU-only build by adding this line at the top of
  `requirements.txt`:
  `--extra-index-url https://download.pytorch.org/whl/cpu`
- **"Gemini API key missing":** the secret name must be exactly `GEMINI_API_KEY`.
- **Gemini 404 / model not found:** set a model your key can use, e.g. `GEMINI_MODEL = "gemini-3.5-flash"` in Secrets.

## RAG concepts used (short glossary)

| Concept | In this project |
|---|---|
| Document parsing | PyMuPDF text per page, plus cleaning of headers, soft hyphens and invisible characters |
| Chunking | ≤500 real tokens with ~100 tokens of overlap, never crossing a page boundary (exact citations) |
| Embedding | 384 numbers describing a text's meaning (`bge-small-en-v1.5`) |
| Cosine similarity | Angle between two vectors. With normalized vectors it equals the dot product. |
| Vector database | FAISS `IndexFlatIP`: exact search over all vectors, held in memory |
| Retrieval | Top 5 chunks, plus the 2 best table chunks (numbers alone embed poorly) |
| Context construction | Numbered sources with page headers: `[Source 1 \| page 18 \| text]` |
| Hallucination prevention | Answers only from sources, a "not in the document" rule, interpretation labelled separately, and every charted number checked against its cited page |
| Evaluation | `RETRIEVAL_CASES` in the tests: does the right page come back for known questions? |
