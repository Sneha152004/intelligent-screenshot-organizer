"""
End-to-End Pipeline & Retrieval Demo Script (3-Layer Storage Architecture)
==========================================================================
Demonstrates the separation of concerns:
Layer 1: Object / File Storage (LocalFileStore)
Layer 2: Application Metadata Storage (SQLiteMetadataStore)
Layer 3: Vector Retrieval Index (ChromaDB)
"""

import json
import os
from pathlib import Path
from src.config import config
from src.embedder import SentenceTransformerEmbedder
from src.pipeline import OCRPipeline
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore
from src.vector_store import VectorStoreManager


def main():
    # Initialize Storage Managers
    file_store = LocalFileStore(storage_dir=config.screenshots_dir)
    metadata_store = SQLiteMetadataStore(db_path=config.metadata_db_path)
    vector_store = VectorStoreManager(
        db_dir=config.chroma_db_dir,
        collection_name=config.collection_name,
    )
    embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)

    # Initialize and run OCR Pipeline
    pipeline = OCRPipeline(
        file_store=file_store,
        metadata_store=metadata_store,
    )
    processed_docs = pipeline.process(metadata_path=config.sample_metadata_path)

    # Extract non-empty chunks and build lightweight metadata for Layer 3 (ChromaDB)
    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []

    for doc in processed_docs:
        sid = doc["screenshot_id"]
        category = doc["metadata"]["category"]

        for chunk_idx, chunk in enumerate(doc["chunks"]):
            cleaned_chunk = chunk.strip()
            if cleaned_chunk:
                doc_id = f"{sid}_chunk_{chunk_idx}"
                chunk_ids.append(doc_id)
                chunk_texts.append(cleaned_chunk)
                chunk_metadatas.append({
                    "screenshot_id": sid,
                    "chunk_index": chunk_idx,
                    "category": category,
                })

    # Generate dense embeddings
    print("\nGenerating embeddings...")
    embeddings = embedder.embed(chunk_texts)

    # Reset ChromaDB collection and index lightweight records
    print("Storing in ChromaDB Vector Store...")
    vector_store.delete_collection()
    vector_store.add_documents(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
    )

    total_screenshots = len(processed_docs)
    total_stored_chunks = vector_store.count_documents()
    total_metadata_records = len(metadata_store.list_all_metadata())

    # Summary Display
    print("\n" + "=" * 50)
    print("INTELLIGENT SCREENSHOT ORGANIZER (3-LAYER ARCHITECTURE)")
    print("=" * 50)
    print(f"Screenshots loaded: {total_screenshots}")
    print(f"\nFile Storage (Layer 1):\n{total_screenshots} screenshots registered in {os.path.abspath(config.screenshots_dir)}")
    print(f"\nStructured Metadata Storage (Layer 2):\n{total_metadata_records} records saved in {os.path.abspath(config.metadata_db_path)}")
    print(f"\nOCR:\n{total_screenshots} screenshots processed")
    print(f"\nChunking:\n{total_stored_chunks} chunks generated")
    print(f"\nEmbeddings:\n{total_stored_chunks} x 384-dimensional vectors")
    print(f"\nChromaDB Vector Index (Layer 3):\n{total_stored_chunks} vector records indexed in '{vector_store.collection_name}'")
    print(f"Database location: {os.path.abspath(config.chroma_db_dir)}")

    # Semantic Retrieval Demonstration
    query_text = "wifi password"
    top_k = config.default_top_k
    print("\n" + "-" * 50)
    print(f"Semantic Query: '{query_text}' (Top-{top_k})")
    print("-" * 50)

    results = pipeline.search_and_enrich(
        query_text=query_text,
        embedder=embedder,
        vector_store=vector_store,
        top_k=top_k,
    )

    for rank, res in enumerate(results, start=1):
        clean_text = res["text"].replace("\n", " | ")
        print(f"\nResult #{rank}:")
        print(f"  ChromaDB Vector Match:")
        print(f"    screenshot_id   = {res['screenshot_id']}")
        print(f"    chunk_index     = {res['chunk_index']}")
        print(f"    text            = {clean_text}")
        print(f"    similarity      = {res['similarity_score']} (distance: {res['distance']:.4f})")
        print(f"  Metadata Resolution (via MetadataStore + FileStore):")
        print(f"    filename        = {res['filename']}")
        print(f"    category        = {res['category']}")
        print(f"    image_uri       = {res['image_uri']}")


if __name__ == "__main__":
    main()
