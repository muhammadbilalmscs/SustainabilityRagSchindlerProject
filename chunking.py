"""
chunking.py - Phase 2 of the RAG pipeline: split page text into small, overlapping chunks.

Responsibility
    Take the list of pages from pdf_processor.py and cut each page into chunks
    of at most ~500 tokens, with ~100 tokens shared between neighbouring
    chunks. Every chunk keeps the page number it came from.

Why chunk at all?
    - An embedding (Phase 3) turns ONE piece of text into ONE vector. A whole
      page about five topics becomes a blurry "average" vector that matches
      none of them well. A smaller chunk is about one thing, so it matches
      questions about that thing much better.
    - Embedding models have an input limit (512 tokens for our model). Text
      beyond the limit is silently cut off and never searchable.
    - We send only the best chunks to the LLM, not the whole 53-page report.

Why chunk inside a page (never across pages)?
    Each chunk then has exactly one page number, so answers can cite pages
    accurately.

How tokens are counted
    A "token" is the unit a model reads: a word, part of a word, or a symbol.
    - Without a tokenizer we ESTIMATE: 1 word = 1.33 tokens (fine for prose).
    - With the embedding model's tokenizer we MEASURE each word exactly.
      This matters: "GRI 305-1" is 2 words but 6 tokens, so pages full of codes
      and numbers are much "longer" than their word count suggests.
    The app always passes the real tokenizer (see embeddings.py).

No library needed here: plain Python lists and loops.

Where the data goes next
    The list of chunk dicts goes to embeddings.py (Phase 3), which turns every
    chunk["text"] into a vector. The chunk_id becomes the chunk's row in the
    FAISS index (Phase 4).
"""

import sys


TOKENS_PER_WORD = 1.33      # rough average for English prose (used only without a tokenizer)
CHUNK_SIZE_TOKENS = 500     # our model's limit is 512; leave room for 2 special tokens
CHUNK_OVERLAP_TOKENS = 100


def count_word_tokens(words, tokenizer=None):
    """
    Return the number of tokens for each word in a list.

    Input:  words     - list of strings, e.g. ["Scope", "305-1", "emissions"]
            tokenizer - the embedding model's tokenizer, or None to estimate
    Output: list of numbers, one per word, e.g. [1, 3, 1]
    """
    if tokenizer is None:
        estimates = []
        for word in words:
            estimates.append(TOKENS_PER_WORD)
        return estimates

    # Tokenize all words in one call (fast). add_special_tokens=False leaves out
    # the start/end markers, which the model adds once per chunk, not per word.
    token_ids_per_word = tokenizer(words, add_special_tokens=False)["input_ids"]
    counts = []
    for token_ids in token_ids_per_word:
        counts.append(len(token_ids))
    return counts


def split_text(text, chunk_size_tokens, overlap_tokens, tokenizer=None):
    """
    Split one text into overlapping pieces of at most chunk_size_tokens tokens.

    Input:  text              - a string (one page of the report)
            chunk_size_tokens - maximum tokens per piece, e.g. 500
            overlap_tokens    - about this many tokens are repeated in the next piece, e.g. 100
            tokenizer         - model tokenizer for exact counts, or None to estimate
    Output: list of strings

    How it works (cutting a page into pieces by whole words):
      1. Add words to the current piece until the next word would go over the size.
      2. Save the piece.
      3. Step back from the end until ~overlap_tokens are covered;
         the next piece starts there, so the boundary text appears in both pieces.
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap must be smaller than the chunk size")

    words = text.split()
    word_tokens = count_word_tokens(words, tokenizer)
    pieces = []
    start = 0

    while start < len(words):
        # 1. Grow the piece word by word.
        end = start
        total_tokens = 0
        while end < len(words) and total_tokens + word_tokens[end] <= chunk_size_tokens:
            total_tokens += word_tokens[end]
            end += 1
        if end == start:  # a single giant "word" bigger than the limit: take it anyway
            end = start + 1

        # 2. Save it.
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break

        # 3. Step back to create the overlap (but always move forward at least one word).
        next_start = end
        overlap_so_far = 0
        while next_start > start + 1 and overlap_so_far + word_tokens[next_start - 1] <= overlap_tokens:
            next_start -= 1
            overlap_so_far += word_tokens[next_start]
        start = next_start

    return pieces


def chunk_pages(pages, chunk_size_tokens=CHUNK_SIZE_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS,
                tokenizer=None):
    """
    Turn the pages from pdf_processor.extract_pages() into chunks with metadata.

    Input:  pages     - list of dicts like {"page_number": 18, "text": "...", "is_empty": False}
            chunk_size_tokens, overlap_tokens - sizes in tokens
            tokenizer - pass model.tokenizer for exact token counts (recommended)
    Output: list of chunk dicts, for example
            {"chunk_id": 31,              # position in the list, later also the FAISS row
             "page_number": 18,           # where the text comes from (for citations)
             "chunk_on_page": 1,          # 0 = first chunk on that page, 1 = second, ...
             "text": "In 2025, Schindler achieved its goal ...",
             "word_count": 360,
             "token_count": 497}          # exact if a tokenizer was given, else estimated
    """
    chunks = []
    for page in pages:
        if page["is_empty"]:
            continue  # nothing useful to search on this page

        pieces = split_text(page["text"], chunk_size_tokens, overlap_tokens, tokenizer)

        for position_on_page, piece in enumerate(pieces):
            words = piece.split()
            token_count = round(sum(count_word_tokens(words, tokenizer)))
            chunks.append({
                "chunk_id": len(chunks),  # 0, 1, 2, ... across the whole document
                "page_number": page["page_number"],
                "chunk_on_page": position_on_page,
                "text": piece,
                "word_count": len(words),
                "token_count": token_count,
            })

    return chunks


def describe_chunks(chunks):
    """
    Summarise a list of chunks: how many, and how big.

    Input:  list of chunk dicts
    Output: dict with the count, smallest/average/largest token count, and pages covered
    """
    token_counts = []
    pages_covered = set()
    for chunk in chunks:
        token_counts.append(chunk["token_count"])
        pages_covered.add(chunk["page_number"])

    return {
        "chunk_count": len(chunks),
        "min_tokens": min(token_counts),
        "avg_tokens": round(sum(token_counts) / len(token_counts)),
        "max_tokens": max(token_counts),
        "pages_covered": len(pages_covered),
    }


# ---------------------------------------------------------------------------
# Quick manual test:  python chunking.py data/schindler_report.pdf
# Uses ESTIMATED tokens, so it runs without the embedding model.
# embeddings.py's test runs the same chunking with EXACT token counts.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pdf_processor import extract_pages

    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python chunking.py path/to/file.pdf")
        sys.exit(1)

    with open(sys.argv[1], "rb") as pdf_file:
        pages = extract_pages(pdf_file.read())

    chunks = chunk_pages(pages)
    stats = describe_chunks(chunks)

    print(f"Settings: <= {CHUNK_SIZE_TOKENS} tokens per chunk, ~{CHUNK_OVERLAP_TOKENS} tokens overlap "
          f"(estimated: 1 word = {TOKENS_PER_WORD} tokens)")
    print(f"Pages:  {len(pages)}  ->  Chunks: {stats['chunk_count']}  (covering {stats['pages_covered']} pages)")
    print(f"Tokens per chunk: min {stats['min_tokens']}, avg {stats['avg_tokens']}, max {stats['max_tokens']}")

    # Show the chunks of page 18 and the words they share (the overlap).
    page_18_chunks = []
    for chunk in chunks:
        if chunk["page_number"] == 18:
            page_18_chunks.append(chunk)

    print(f"\nPage 18 has {len(page_18_chunks)} chunks:")
    for chunk in page_18_chunks:
        print(f"  chunk_id={chunk['chunk_id']}  chunk_on_page={chunk['chunk_on_page']}  "
              f"words={chunk['word_count']}  tokens={chunk['token_count']}")

    if len(page_18_chunks) >= 2:
        first_text = page_18_chunks[0]["text"]
        second_start = " ".join(page_18_chunks[1]["text"].split()[:12])
        print("\nStart of chunk 1:  ", second_start, "...")
        print("Also found at the end of chunk 0?", second_start in first_text)

    # Simple keyword lookup - a preview of what semantic search will do much better.
    keyword = "renewable electricity"
    print(f"\nChunks containing '{keyword}':")
    for chunk in chunks:
        if keyword in chunk["text"].lower():
            print(f"  chunk_id={chunk['chunk_id']:3d}  page {chunk['page_number']}")
