"""
embeddings.py - Phase 3 of the RAG pipeline: turn text into vectors (embeddings).

Responsibility
    Load a small embedding model once, and use it to convert
      - every chunk  -> one vector   (done once, when the PDF is processed)
      - the question -> one vector   (done for every question)

What is an embedding?
    A list of numbers (here 384 of them) that describes the MEANING of a text.
    The model was trained so that texts with similar meaning get similar
    vectors, even when they share no words:
        "renewable electricity"  ->  [0.02, -0.07, 0.11, ...]
        "green power"            ->  [0.03, -0.06, 0.10, ...]   (close!)
        "child labor policy"     ->  [-0.09, 0.04, -0.01, ...]  (far away)

Why Sentence Transformers and BAAI/bge-small-en-v1.5?
    - sentence-transformers wraps Hugging Face models in a single .encode() call.
    - bge-small is small (~130 MB, runs on a laptop CPU, no API key, free)
      and is strong at retrieval tasks.
    - It reads up to 512 tokens, which fits our ~500-token chunks. The popular
      all-MiniLM-L6-v2 stops at 256 tokens and would silently cut every chunk in half.

Where the data goes next
    The matrix of chunk vectors goes into the FAISS index (Phase 4).
    The question vector is what FAISS searches with.
"""

import sys

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models were trained to see this sentence in front of SHORT QUESTIONS, but
# not in front of the document chunks. It tells the model "this is a search
# query, find passages that answer it", which improves retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_model(model_name=MODEL_NAME):
    """
    Load the embedding model. The first call downloads it (~130 MB) from
    Hugging Face; later calls load it from the local cache.

    Input:  model name (str)
    Output: a SentenceTransformer model object

    Loading takes a few seconds, so do it ONCE and reuse the model.
    (In the Streamlit app we will wrap this with @st.cache_resource.)
    """
    return SentenceTransformer(model_name)


def get_model_info(model):
    """
    Describe the model, for the debug section of the app.

    Output: {"name": ..., "dimensions": 384, "max_tokens": 512}
    """
    return {
        "name": MODEL_NAME,
        "dimensions": model.get_embedding_dimension(),
        "max_tokens": model.max_seq_length,
    }


def count_tokens(model, text):
    """
    Count tokens exactly as the embedding model sees them.

    Input:  model, text (str)
    Output: number of tokens (int), including the 2 special start/end tokens

    Example: "Natural gas 65 671 MWh" -> about 11 tokens, because numbers and
    units are split into several pieces ("65", "67", "##1", "mw", "##h").
    """
    return len(model.tokenizer(text)["input_ids"])


def embed_chunks(model, chunks):
    """
    Convert every chunk's text into a vector.

    Input:  model, chunks (list of dicts from chunking.chunk_pages)
    Output: numpy array with shape (number_of_chunks, 384), dtype float32
            Row i is the vector for chunks[i] - same order, so chunk_id == row number.

    normalize_embeddings=True scales every vector to length 1. Then the cosine
    similarity of two vectors is simply their dot product, which is exactly
    what our FAISS index will compute in Phase 4.
    """
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])

    vectors = model.encode(
        texts,
        batch_size=16,              # embed 16 chunks at a time
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")  # FAISS requires float32


def embed_query(model, question):
    """
    Convert one user question into a vector.

    Input:  model, question (str), e.g. "How much green power does Schindler use?"
    Output: numpy array with shape (1, 384), dtype float32
            (a 1-row matrix, because FAISS always searches with a matrix of queries)
    """
    vector = model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vector, dtype="float32")


def cosine_similarity(vector_a, vector_b):
    """
    Measure how similar two vectors are: 1.0 = same direction (same meaning),
    0 = unrelated, negative = opposite. Written out by hand so you can see the formula:

        cosine = (a · b) / (|a| * |b|)

    For normalized vectors |a| = |b| = 1, so cosine is just the dot product a · b.
    FAISS will do this for us in Phase 4; this function is here for learning.
    """
    dot_product = np.dot(vector_a, vector_b)
    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)
    return float(dot_product / (length_a * length_b))


# ---------------------------------------------------------------------------
# Quick manual test:  python embeddings.py data/schindler_report.pdf
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from chunking import chunk_pages
    from pdf_processor import extract_pages

    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python embeddings.py path/to/file.pdf")
        sys.exit(1)

    print("Loading model ...")
    model = load_model()
    info = get_model_info(model)
    print(f"Model: {info['name']}  |  {info['dimensions']} dimensions  |  max {info['max_tokens']} tokens")

    # --- 1. Similarity between short sentences (no PDF needed) -------------
    print("\n1) Meaning, not words:")
    sentences = [
        "The company uses renewable electricity.",
        "Schindler buys green power for its offices.",
        "Employees receive safety training.",
    ]
    sentence_vectors = model.encode(sentences, normalize_embeddings=True)
    print(f"   '{sentences[0]}'  vs")
    for i in [1, 2]:
        score = cosine_similarity(sentence_vectors[0], sentence_vectors[i])
        print(f"     {score:.3f}  '{sentences[i]}'")

    # --- 2. Embed all chunks of the report ---------------------------------
    with open(sys.argv[1], "rb") as pdf_file:
        pages = extract_pages(pdf_file.read())
    # Pass the model's tokenizer so chunk sizes are measured in REAL tokens.
    chunks = chunk_pages(pages, tokenizer=model.tokenizer)

    start_time = time.time()
    chunk_vectors = embed_chunks(model, chunks)
    print(f"\n2) Embedded {len(chunks)} chunks in {time.time() - start_time:.1f} s")
    print(f"   Matrix shape: {chunk_vectors.shape}  (rows = chunks, columns = dimensions)")
    print(f"   Vector length of chunk 0: {np.linalg.norm(chunk_vectors[0]):.3f}  (1.000 = normalized)")
    print(f"   First 5 numbers of chunk 0: {np.round(chunk_vectors[0][:5], 3)}")

    # --- 3. Do the chunks fit in the model's 512-token window? -------------
    # count_tokens() tokenizes the WHOLE chunk again (with start/end markers),
    # an independent check of what chunking.py measured word by word.
    token_counts = []
    for chunk in chunks:
        token_counts.append(count_tokens(model, chunk["text"]))
    too_long = []
    for chunk, tokens in zip(chunks, token_counts):
        if tokens > info["max_tokens"]:
            too_long.append((chunk["chunk_id"], chunk["page_number"], tokens))
    print(f"\n3) Real token counts: min {min(token_counts)}, avg {sum(token_counts) // len(token_counts)}, "
          f"max {max(token_counts)}  (limit {info['max_tokens']})")
    print(f"   Chunks over the limit: {too_long if too_long else 'none'}")

    # --- 4. A first semantic search (FAISS will do this properly in Phase 4) ---
    question = "How much green power does the company use?"
    question_vector = embed_query(model, question)
    # Dot product of the question with ALL chunk vectors at once = cosine scores.
    scores = chunk_vectors @ question_vector[0]
    best_rows = np.argsort(scores)[::-1][:3]  # indexes of the 3 highest scores
    print(f"\n4) Question: '{question}'  (note: the word 'green' is not needed in the text)")
    for row in best_rows:
        chunk = chunks[row]
        print(f"   score {scores[row]:.3f}  page {chunk['page_number']:2d}  {chunk['text'][:80]}...")
