"""
Vector Store Module (Layer 3: ChromaDB Vector Index & Retrieval Layer)
======================================================================
ChromaDB is utilized exclusively as a lightweight vector retrieval index.
It stores ONLY chunk embeddings, chunk text, and minimal retrieval metadata
(screenshot_id, chunk_index, category). Raw image bytes, canonical file paths,
and full application metadata belong to Layers 1 and 2.
"""

import os
from typing import Any, Dict, List, Optional, Union
import chromadb


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store operations for chunk vector retrieval.
    """

    def __init__(
        self,
        db_dir: str = "chroma_db",
        collection_name: str = "screenshot_chunks",
    ):
        """
        Initialize VectorStoreManager with persistent database directory and collection.

        Args:
            db_dir: Path to directory for persistent ChromaDB storage.
            collection_name: Name of the ChromaDB collection.
        """
        self.db_dir = os.path.abspath(db_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.create_collection()

    def create_collection(self):
        """
        Retrieves or creates the ChromaDB collection using cosine similarity metric.

        Returns:
            ChromaDB collection instance.
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Inserts document text chunks, embedding vectors, and lightweight metadata into ChromaDB.

        Args:
            ids: List of unique document chunk IDs (e.g. "ss_001_chunk_0").
            documents: List of text chunk strings.
            embeddings: List of embedding float vectors.
            metadatas: List of lightweight metadata dicts (screenshot_id, chunk_index, category).
        """
        if not ids or not documents or not embeddings or not metadatas:
            return

        if len(ids) != len(documents) or len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError("Length mismatch between ids, documents, embeddings, and metadatas.")

        # Ensure metadata stored in ChromaDB remains minimal and clean
        sanitized_metadatas: List[Dict[str, Union[str, int, float, bool]]] = []
        for meta in metadatas:
            sanitized: Dict[str, Union[str, int, float, bool]] = {
                "screenshot_id": str(meta.get("screenshot_id", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "category": str(meta.get("category", "General")),
            }
            sanitized_metadatas.append(sanitized)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=sanitized_metadatas,
        )

    def count_documents(self) -> int:
        """
        Returns the total number of document chunk records stored in the collection.

        Returns:
            Total document count integer.
        """
        return self.collection.count()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single stored document record by its unique ID.

        Args:
            doc_id: Unique record ID string (e.g. "ss_001_chunk_0").

        Returns:
            Dictionary containing id, document, embedding, and lightweight metadata if found, else None.
        """
        res = self.collection.get(
            ids=[doc_id],
            include=["documents", "embeddings", "metadatas"],
        )

        if res and res.get("ids") and len(res["ids"]) > 0:
            return {
                "id": res["ids"][0],
                "document": res["documents"][0] if res.get("documents") else None,
                "embedding": (
                    res["embeddings"][0]
                    if res.get("embeddings") is not None and len(res["embeddings"]) > 0
                    else None
                ),
                "metadata": res["metadatas"][0] if res.get("metadatas") else None,
            }
        return None

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector similarity search against stored chunk embeddings.

        Args:
            query_embedding: Dense float vector representation of the search query.
            top_k: Number of top matches to retrieve.
            where: Optional metadata filter dictionary.

        Returns:
            List of result dictionaries containing id, document, metadata, distance, and similarity score.
        """
        if self.count_documents() == 0:
            return []

        # Bound top_k to total available documents
        actual_k = min(top_k, self.count_documents())

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids)):
                dist = float(distances[i]) if i < len(distances) else 0.0
                # Cosine distance ranges from 0 (identical) to 2. Calculate similarity score:
                sim_score = max(0.0, round(1.0 - dist, 4))
                meta = metadatas[i] if i < len(metadatas) else {}

                formatted_results.append(
                    {
                        "id": ids[i],
                        "document": documents[i] if i < len(documents) else "",
                        "metadata": meta,
                        "distance": dist,
                        "similarity_score": sim_score,
                        "screenshot_id": meta.get("screenshot_id", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "category": meta.get("category", ""),
                    }
                )

        return formatted_results

    def delete_collection(self) -> None:
        """
        Deletes the ChromaDB collection and resets internal state.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.create_collection()
