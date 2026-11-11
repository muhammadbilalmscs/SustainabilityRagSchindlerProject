"""
vector_store.py - Phase 4 of the RAG pipeline: store vectors in FAISS and search them.

Responsibility
    - build_index(): put all chunk vectors into a FAISS index
    - search():      find the chunks whose vectors are closest to a question vector

What is a vector database?
    A store that answers one question very fast: "which stored vectors are most
    similar to this new vector?" A normal database finds rows by exact values
    (WHERE page = 18). A vector database finds rows by closeness in meaning.

Why FAISS?
    FAISS (Facebook AI Similarity Search) is a free library, runs in memory,
    and needs no server. For 100 chunks we could loop in numpy ourselves
    (Phase 3 did exactly that), but FAISS scales to millions and is the
    standard tool you will meet in real projects.

Which index type?
    IndexFlatIP = "flat" (compare against every vector, exact results)
                + "IP"   (inner product = dot product).
    Our vectors are normalized to length 1, so inner product == cosine similarity.
    Flat is perfect for small documents; approximate indexes only pay off at
    hundreds of thousands of vectors.

Metadata lives separately
    FAISS stores ONLY numbers. It returns row positions (e.g. 26) and scores.
    We keep the text and page numbers in our Python `chunks` list, where
    chunks[26] is the chunk behind FAISS row 26.

Where the data goes next
    search() returns the top chunks with their scores to rag_pipeline.py,
    which builds the LLM context from them.

No disk storage
    The index lives in memory only. On Streamlit Community Cloud the disk is
    temporary, so the app simply rebuilds the index from the uploaded PDF
    (it takes about 15 seconds) instead of saving FAISS files.
"""

import faiss
import numpy as np


def build_index(vectors):
    """
    Create a FAISS index and add all chunk vectors to it.

    Input:  vectors - numpy float32 array, shape (number_of_chunks, 384), normalized
    Output: a faiss.IndexFlatIP holding all vectors; row i belongs to chunks[i]
    """
    dimensions = vectors.shape[1]          # 384 for bge-small
    index = faiss.IndexFlatIP(dimensions)  # exact search by inner product (= cosine here)
    index.add(vectors)                     # rows are numbered 0, 1, 2, ... in the order added
    return index


def search(index, chunks, query_vector, top_k=5):
    """
    Find the top_k chunks most similar to the question.

    Input:  index        - the FAISS index from build_index()
            chunks       - the list of chunk dicts, in the same order as the vectors
            query_vector - numpy float32 array, shape (1, 384), from embeddings.embed_query()
            top_k        - how many results to return
    Output: list of dicts, best first, for example
            [{"rank": 1, "score": 0.703, "chunk_id": 26, "page_number": 18,
              "text": "...", "source": "text"}, ...]
    """
    top_k = min(top_k, index.ntotal)  # cannot return more results than we stored

    # FAISS searches a whole batch of questions at once, so it returns 2-D arrays:
    #   scores[0]  = similarity scores for our one question, best first
    #   row_ids[0] = the matching row positions
    scores, row_ids = index.search(query_vector, top_k)

    results = []
    for rank, (score, row_id) in enumerate(zip(scores[0], row_ids[0]), start=1):
        if row_id == -1:  # FAISS uses -1 for "no result"
            continue
        chunk = chunks[row_id]          # metadata lookup: row number -> chunk dict
        result = dict(chunk)            # copy, so we do not change the original chunk
        result["rank"] = rank
        result["score"] = round(float(score), 4)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Quick manual test:  python vector_store.py data/schindler_report.pdf
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from chunking import chunk_pages
    from embeddings import embed_chunks, embed_query, load_model
    from pdf_processor import extract_pages

    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py path/to/file.pdf")
        sys.exit(1)

    model = load_model()
    with open(sys.argv[1], "rb") as pdf_file:
        pages = extract_pages(pdf_file.read())
    chunks = chunk_pages(pages, tokenizer=model.tokenizer)
    vectors = embed_chunks(model, chunks)

    index = build_index(vectors)
    print(f"FAISS index: {index.ntotal} vectors of {index.d} dimensions")

    # Sanity check: FAISS scores must equal our hand-made numpy dot product from Phase 3.
    question = "How does Schindler prevent child labor in its supply chain?"
    query_vector = embed_query(model, question)
    results = search(index, chunks, query_vector, top_k=5)
    numpy_best = float(np.max(vectors @ query_vector[0]))
    print(f"Best FAISS score {results[0]['score']:.4f} == numpy score {numpy_best:.4f}")

    print(f"\nQuestion: {question}")
    for result in results:
        print(f"  #{result['rank']}  score {result['score']:.3f}  page {result['page_number']:2d}  "
              f"{result['text'][:70]}...")
