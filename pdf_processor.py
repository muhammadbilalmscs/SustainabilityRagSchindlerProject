"""
pdf_processor.py - Phase 1 of the RAG pipeline: turn a PDF into clean text, page by page.

Responsibility
    Take the raw bytes of a PDF and return a list of pages. Each page is a small
    dictionary holding its page number and its cleaned text.

Why PyMuPDF?
    It is fast (a 50-page report takes well under a second), it keeps the
    natural reading order of the text, and it can open a PDF straight from
    memory. Streamlit gives us uploaded files as bytes, so we never need to
    save the PDF to disk - useful on Streamlit Community Cloud.

Where the data goes next
    The list of pages goes to chunking.py (Phase 2). The page_number stored
    here is what lets the final answer say "see page 18".
"""

import sys

import pymupdf  # PyMuPDF. Older tutorials write "import fitz" - same library, old name.


# Pages with fewer characters than this are treated as "empty".
# A page that only holds a page number or a logo is not useful for answering questions.
MIN_CHARS_PER_PAGE = 50


def clean_text(text):
    """
    Remove invisible characters and extra spaces from text extracted from a PDF.

    Input:  raw text from one page (str)
    Output: cleaned text (str)

    Example: "purchas\\xad\\ning  EACs " -> "purchasing EACs"
    """
    # ﻿ is an invisible "byte order mark". This report puts one after "Schindler" in every page header.
    text = text.replace("﻿", "")
    # \xad is a "soft hyphen": an invisible mark that shows where a word may be split across lines.
    # Here it sits at a line end: "purchas\xad\ning". Removing the hyphen AND the line break
    # joins the word back together; otherwise "purchasing" would never match a search.
    text = text.replace("\xad\n", "")
    text = text.replace("\xad", "")
    #   is an "en space" used in headings such as "1 General information".
    text = text.replace(" ", " ")

    cleaned_lines = []
    for line in text.splitlines():
        # " ".join(line.split()) collapses runs of spaces into one space and trims both ends.
        line = " ".join(line.split())
        if line:  # skip blank lines
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_repeated_lines(pages, min_share=0.5):
    """
    Remove header/footer lines that repeat on many pages.

    Every page of this report starts with "Schindler" and "Nonfinancial Report 2025".
    Those lines carry no meaning, and if we keep them they appear in every chunk
    and make different chunks look more similar than they really are.

    Input:  list of page dicts (as built by extract_pages)
            min_share: a line is "repeated" if it appears on at least this share of pages
    Output: the same list, with the repeated lines removed from each page's text
    """
    # Step 1: count on how many pages each line appears.
    line_counts = {}
    for page in pages:
        unique_lines_on_page = set(page["text"].splitlines())
        for line in unique_lines_on_page:
            line_counts[line] = line_counts.get(line, 0) + 1

    # Step 2: lines that appear on many pages are headers/footers.
    min_pages = max(2, int(len(pages) * min_share))
    repeated = set()
    for line, count in line_counts.items():
        if count >= min_pages:
            repeated.add(line)

    # Step 3: rebuild each page's text without those lines.
    for page in pages:
        kept_lines = []
        for line in page["text"].splitlines():
            if line not in repeated:
                kept_lines.append(line)
        page["text"] = "\n".join(kept_lines)
        page["char_count"] = len(page["text"])
        page["is_empty"] = page["char_count"] < MIN_CHARS_PER_PAGE

    return pages


def extract_pages(pdf_bytes):
    """
    Extract the text of every page in a PDF.

    Input:  pdf_bytes - the whole PDF file as bytes
            (from open(path, "rb").read() or from Streamlit's uploaded_file.getvalue())
    Output: a list with one dict per page, for example:
            {"page_number": 18, "text": "Transitioning to renewable ...",
             "char_count": 3850, "is_empty": False}

    page_number is the PDF page number (1 = first page of the file), which is
    what a PDF viewer shows. Note: this report prints its own page numbers one
    lower (PDF page 18 shows "17" at the top) because the cover page is not numbered.

    Raises ValueError if the file is not a readable PDF.
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError(f"Could not open the file as a PDF: {error}")

    if document.is_encrypted:
        document.close()
        raise ValueError("This PDF is password-protected. Please upload an unprotected copy.")

    pages = []
    # enumerate(..., start=1) gives page numbers 1, 2, 3... instead of 0, 1, 2...
    for page_number, page in enumerate(document, start=1):
        try:
            raw_text = page.get_text()
        except Exception as error:
            # One broken page should not stop the whole document.
            print(f"Warning: could not read page {page_number}: {error}")
            raw_text = ""

        text = clean_text(raw_text)
        pages.append({
            "page_number": page_number,
            "text": text,
            "char_count": len(text),
            "is_empty": len(text) < MIN_CHARS_PER_PAGE,
        })

    document.close()

    pages = remove_repeated_lines(pages)
    return pages


def get_document_info(pdf_bytes):
    """
    Read basic facts about a PDF: title, page count and how many pages have text.

    Input:  pdf_bytes - the PDF file as bytes
    Output: dict, for example
            {"title": "...", "page_count": 53, "text_pages": 53, "empty_pages": [],
             "likely_scanned": False}

    "likely_scanned" is True when most pages have no text. A scanned PDF is just
    pictures of pages, so it needs OCR (optical character recognition) first.
    """
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    title = document.metadata.get("title") or "Untitled document"
    document.close()

    pages = extract_pages(pdf_bytes)

    empty_pages = []
    for page in pages:
        if page["is_empty"]:
            empty_pages.append(page["page_number"])

    page_count = len(pages)
    text_pages = page_count - len(empty_pages)

    return {
        "title": title,
        "page_count": page_count,
        "text_pages": text_pages,
        "empty_pages": empty_pages,
        "likely_scanned": text_pages < page_count / 2,
    }


# ---------------------------------------------------------------------------
# Quick manual test:  python pdf_processor.py data/schindler_report.pdf
# This block only runs when you start this file directly, not when another
# file does "import pdf_processor".
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Windows terminals sometimes cannot print characters such as "CO₂".
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py path/to/file.pdf")
        sys.exit(1)

    with open(sys.argv[1], "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    info = get_document_info(pdf_bytes)
    print("Title:        ", info["title"])
    print("Pages:        ", info["page_count"])
    print("Pages w/ text:", info["text_pages"])
    print("Empty pages:  ", info["empty_pages"] or "none")
    print("Scanned?      ", "probably yes - needs OCR" if info["likely_scanned"] else "no")

    pages = extract_pages(pdf_bytes)
    total_chars = 0
    for page in pages:
        total_chars += page["char_count"]
    print("Total chars:  ", total_chars)

    # Show the start of one page so you can see what the cleaned text looks like.
    sample = pages[17]  # index 17 = PDF page 18
    print(f"\n--- Page {sample['page_number']} (first 600 characters) ---")
    print(sample["text"][:600])
