"""
table_extractor.py - Phase 7: find tables in the PDF and make them searchable.

Responsibility
    - extract_tables(): find every table, return each as a pandas DataFrame
    - tables_to_chunks(): turn tables into text chunks (Markdown rows) so the
      normal search can find them and Gemini can read the exact figures

Why PyMuPDF and not pdfplumber?
    The plan said pdfplumber, so we tested it first (Phase 1 inspection).
    The Schindler tables have no ruling lines, only white space between the
    columns. pdfplumber's default (line-based) finder found 1 table in the
    whole report, and its text-based mode cut words in half ("Key fig|ures").
    PyMuPDF's page.find_tables() found 18 usable tables, such as energy
    in MWh, CO2e emissions by scope, and waste in tonnes, with columns for
    2025 / 2024 / 2020. So we use PyMuPDF, which we already depend on.

Why read each cell with get_textbox()?
    find_tables() locates the cells correctly, but its own text extraction drops
    spaces ("Naturalgas"). Reading the text inside each cell's rectangle
    gives clean labels ("Natural gas").

Known limits (we do not hide them)
    - Footnote markers stick to values: "100%" with footnote 2 reads "100%2".
    - A few cells are merged wrongly, e.g. "7 124 197 12 370 288" (two numbers).
    - Numbers inside bar charts are drawn graphics, not tables, so they are not here
      (their labels are still in the page text).
    parse_number() refuses ambiguous values instead of guessing.

Where the data goes next
    - Table chunks are embedded and indexed together with text chunks (rag_pipeline.py).
    - The DataFrames are shown in the app's "Extracted tables" section.
"""

import re

import pandas as pd
import pymupdf

from chunking import CHUNK_SIZE_TOKENS, count_word_tokens


def read_table_rows(page, table):
    """
    Read the text of every cell of one table, row by row.

    Input:  page (pymupdf page), table (from page.find_tables())
    Output: list of rows, each a list of strings, e.g. [["Natural gas", "65 671", "69 336"], ...]
    """
    rows = []
    for row in table.rows:
        cells = []
        for cell_box in row.cells:
            if cell_box is None:  # a merged cell has no box of its own
                cells.append("")
            else:
                text = page.get_textbox(pymupdf.Rect(cell_box))
                cells.append(" ".join(text.split()))  # collapse line breaks and spaces
        rows.append(cells)
    return rows


def rows_to_dataframe(rows):
    """
    Turn raw rows into a tidy DataFrame: the first row becomes the column names,
    and completely empty rows and columns are removed.

    Input:  list of rows (lists of strings)
    Output: pandas DataFrame, or None if nothing useful is left
    """
    df = pd.DataFrame(rows)
    df = df.replace("", None)
    df = df.dropna(how="all")            # drop rows where every cell is empty
    df = df.dropna(axis=1, how="all")    # drop columns where every cell is empty
    if df.shape[0] < 2 or df.shape[1] < 2:
        return None                      # too small to be a real table

    header = []
    for position, name in enumerate(df.iloc[0]):
        if pd.isna(name) or name in header:
            name = f"column_{position + 1}"  # column names must exist and be unique
        header.append(name)

    df = df.iloc[1:].copy()
    df.columns = header
    df = df.fillna("")
    return df.reset_index(drop=True)


def read_heading(page, table, band_height=45):
    """
    Read the text printed just above a table, which usually holds its title,
    e.g. "Greenhouse gas (GHG) emissions overview: scope 1, 2, and 3 / t CO₂e / 2025 ...".

    Why: a table chunk is mostly numbers, and numbers carry little meaning for an
    embedding model. Without the title, a question about "scope 1 emissions"
    did not find the scope 1 table at all. The title gives the chunk its topic.

    Input:  page, table, band_height (points above the table to read; 72 points = 1 inch)
    Output: one line of text (may contain a stray line from above - good enough for search)
    """
    box = pymupdf.Rect(table.bbox)
    band = pymupdf.Rect(box.x0, box.y0 - band_height, box.x1, box.y0)
    lines = []
    for line in page.get_textbox(band).splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return " / ".join(lines)[:300]


def number_heavy_pages(pages, min_share=0.15):
    """
    Pick the pages worth searching for tables: pages where many text lines are numbers.

    Why: find_tables() is slow, 2-6 seconds PER PAGE (about 150 s for this report),
    and most pages are plain prose. In the Schindler report the key-figure pages
    have 50-63% numeric lines, while prose pages have about 3%.

    Input:  pages from pdf_processor.extract_pages(), min_share (0.15 = 15% of lines)
    Output: list of page numbers, e.g. [1, 30, 36, 37, 38, 39, 48]
    """
    selected = []
    for page in pages:
        lines = page["text"].splitlines()
        if not lines:
            continue
        numeric_lines = 0
        for line in lines:
            parts = line.split()
            if parts and all(parse_number(part) is not None for part in parts):
                numeric_lines += 1  # e.g. "65 671 69 336" or "2.7"
        if numeric_lines / len(lines) >= min_share:
            selected.append(page["page_number"])
    return selected


def extract_tables(pdf_bytes, page_numbers=None):
    """
    Find tables in a PDF.

    Input:  pdf_bytes    - the PDF file as bytes
            page_numbers - only search these pages (see number_heavy_pages);
                           None = search every page (slow: ~3 s per page)
    Output: list of dicts, for example
            {"table_id": 4, "page_number": 37, "table_on_page": 1, "dataframe": <DataFrame>,
             "numeric_share": 0.74, "heading": "Greenhouse gas (GHG) emissions overview ..."}
    """
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    tables = []

    for page_number, page in enumerate(document, start=1):
        if page_numbers is not None and page_number not in page_numbers:
            continue
        try:
            found = page.find_tables().tables
        except Exception as error:  # one bad page should not stop the rest
            print(f"Warning: table search failed on page {page_number}: {error}")
            continue

        for table_on_page, table in enumerate(found, start=1):
            df = rows_to_dataframe(read_table_rows(page, table))
            if df is not None:
                tables.append({
                    "table_id": len(tables),
                    "page_number": page_number,
                    "table_on_page": table_on_page,
                    "dataframe": df,
                    "numeric_share": numeric_share(df),
                    "heading": read_heading(page, table),
                })

    document.close()
    return tables


def numeric_share(df):
    """
    Share of non-empty cells that are clean numbers (0.0 to 1.0).

    Key-figure tables score high (most cells are values). "Tables" that are
    really text laid out in columns (risk descriptions, the GRI index) score low.
    """
    filled = 0
    numbers = 0
    for row in df.itertuples(index=False):
        for value in row:
            if str(value).strip():
                filled += 1
                if parse_number(value) is not None:
                    numbers += 1
    if filled == 0:
        return 0.0
    return round(numbers / filled, 2)


# Tables below this share are text laid out in columns. Their words are already
# in the page text chunks, so turning them into chunks too would only add duplicates.
MIN_NUMERIC_SHARE = 0.3


def row_to_markdown(values):
    """["Natural gas", "65 671"] -> "| Natural gas | 65 671 |" """
    cells = []
    for value in values:
        cells.append(str(value))
    return "| " + " | ".join(cells) + " |"


def tables_to_chunks(tables, first_chunk_id, tokenizer=None, max_tokens=CHUNK_SIZE_TOKENS):
    """
    Turn tables into text chunks that look like the normal text chunks.

    Each chunk is a Markdown table: a title line, the header row, then data rows.
    A long table is split into several chunks, and every part repeats the header,
    so "65 671" is never separated from "2025" and "MWh".

    Input:  tables         - from extract_tables()
            first_chunk_id - id for the first table chunk (= number of text chunks),
                             so ids stay equal to FAISS row numbers
            tokenizer      - model tokenizer for exact sizes (or None to estimate)
    Output: list of chunk dicts with "source": "table"
    """
    chunks = []
    for table in tables:
        if table["numeric_share"] < MIN_NUMERIC_SHARE:
            continue  # a text table: already searchable through the page text
        df = table["dataframe"]
        title = f"Table from page {table['page_number']}: {table['heading']}"
        header = row_to_markdown(df.columns)
        fixed_part = title + "\n" + header
        fixed_tokens = sum(count_word_tokens(fixed_part.split(), tokenizer))

        current_rows = []
        current_tokens = fixed_tokens
        all_rows = []
        for values in df.itertuples(index=False):
            all_rows.append(row_to_markdown(values))

        for row_text in all_rows:
            row_tokens = sum(count_word_tokens(row_text.split(), tokenizer))
            # Start a new chunk when this row would not fit any more.
            if current_rows and current_tokens + row_tokens > max_tokens:
                chunks.append(make_table_chunk(fixed_part, current_rows, table, first_chunk_id + len(chunks)))
                current_rows = []
                current_tokens = fixed_tokens
            current_rows.append(row_text)
            current_tokens += row_tokens

        if current_rows:
            chunks.append(make_table_chunk(fixed_part, current_rows, table, first_chunk_id + len(chunks)))

    return chunks


def make_table_chunk(fixed_part, rows, table, chunk_id):
    """Build one table chunk dict (same keys as text chunks, plus source and table_id)."""
    text = fixed_part + "\n" + "\n".join(rows)
    return {
        "chunk_id": chunk_id,
        "page_number": table["page_number"],
        "chunk_on_page": table["table_on_page"],
        "text": text,
        "word_count": len(text.split()),
        "token_count": None,  # filled in by the pipeline if needed; not used for search
        "source": "table",
        "table_id": table["table_id"],
    }


# A number as printed in this report: optional minus (– or -), digit groups
# separated by spaces or commas ("65 671", "1,440"), optional decimals, optional %.
NUMBER_PATTERN = re.compile(r"^[–-]?\d{1,3}([ ,]\d{3})*(\.\d+)?%?$|^[–-]?\d+(\.\d+)?%?$")


def parse_number(text):
    """
    Convert a printed number into a float, or return None if it is not one clean number.

    "65 671" -> 65671.0     "–31%" -> -31.0     "2.7" -> 2.7
    "7 124 197 12 370 288" -> None   (two numbers merged: refuse instead of guessing)
    """
    text = str(text).strip()
    if not NUMBER_PATTERN.match(text):
        return None
    cleaned = text.replace("–", "-").replace(" ", "").replace(",", "").replace("%", "")
    return float(cleaned)


# ---------------------------------------------------------------------------
# Quick manual test:  python table_extractor.py data/schindler_report.pdf
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py path/to/file.pdf")
        sys.exit(1)

    with open(sys.argv[1], "rb") as pdf_file:
        tables = extract_tables(pdf_file.read())

    pages_with_tables = sorted({table["page_number"] for table in tables})
    print(f"Found {len(tables)} tables on pages {pages_with_tables}")

    for table in tables:
        if table["page_number"] == 37:
            print(f"\n--- Table {table['table_id']} (page 37) ---")
            print(table["dataframe"].head(6).to_string())

    print("\nNumeric share per table (>= 0.3 becomes a searchable table chunk):")
    for table in tables:
        print(f"  table {table['table_id']:2d}  page {table['page_number']:2d}  "
              f"rows {len(table['dataframe']):2d}  numeric {table['numeric_share']:.2f}")

    table_chunks = tables_to_chunks(tables, first_chunk_id=0)
    print(f"\n{len(table_chunks)} table chunks. The first one:")
    print(table_chunks[0]["text"][:600])

    print("\nparse_number checks:")
    for sample in ["65 671", "–31%", "2.7", "7 124 197 12 370 288", "100%2"]:
        print(f"  {sample!r:>24} -> {parse_number(sample)}")
