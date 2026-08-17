"""
Integration & Verification Script for OCR + Embedding Pipeline
==============================================================
Runs the mock OCR pipeline on the screenshot dataset, generates embeddings for all
extracted chunks, verifies consistency, and demonstrates screenshot-to-embedding traceability.
"""

import json
from pathlib import Path
from src.pipeline import OCRPipeline
from src.embedder import SentenceTransformerEmbedder


def main():
    # Step 1: Initialize pipeline and process dataset
    pipeline = OCRPipeline()
    processed_docs = pipeline.process(metadata_path="sample/metadata.csv")

    # Step 2: Collect non-empty chunks and pair each chunk with its screenshot filename
    chunk_traceability = []  # List of tuples: (filename, chunk_text)
    all_chunks = []

    for doc in processed_docs:
        filename = Path(doc["image_path"]).name
        for chunk in doc["chunks"]:
            cleaned_chunk = chunk.strip()
            if cleaned_chunk:  # Only non-empty chunks
                chunk_traceability.append((filename, cleaned_chunk))
                all_chunks.append(cleaned_chunk)

    # Step 3: Generate embeddings
    print("\nGenerating embeddings...")
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed(all_chunks)

    # Step 4: Verify consistency dynamically
    total_chunks = len(all_chunks)
    total_embeddings = len(embeddings)

    if total_chunks != total_embeddings:
        raise ValueError(
            f"Mismatch between chunk count ({total_chunks}) and embedding count ({total_embeddings})."
        )

    if total_embeddings == 0:
        embedding_dim = 0
    else:
        embedding_dim = len(embeddings[0])
        for idx, emb in enumerate(embeddings):
            if len(emb) != embedding_dim:
                raise ValueError(
                    f"Embedding at index {idx} has dimension {len(emb)}, expected {embedding_dim}."
                )

    print(f"Total chunks: {total_chunks}")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Embedding dimension: {embedding_dim}")

    # Step 5: Demonstrate screenshot traceability for the first few chunks
    print("\nDemonstrating Screenshot Traceability (First 5 Chunks):")
    print("-" * 55)
    sample_count = min(5, len(chunk_traceability))
    for i in range(sample_count):
        filename, chunk_text = chunk_traceability[i]
        # Replace newlines in display chunk for clean output formatting
        display_chunk = chunk_text.replace("\n", " | ")
        print(f"{filename}")
        print(f"  Chunk: {display_chunk}")
        print(f"  Embedding dimension: {len(embeddings[i])}\n")

    # Step 6: Display sample processed document structure
    print("=" * 55)
    print("Sample Processed Document Output (First Document):")
    print("=" * 55)
    print(json.dumps(processed_docs[0], indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
