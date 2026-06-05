from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._chroma_client = chromadb.EphemeralClient()
            self._collection = self._chroma_client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata) if doc.metadata else {}
        metadata['doc_id'] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
        query_emb = self._embedding_fn(query)
        results = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            results.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            if self._use_chroma and self._collection is not None:
                self._collection.add(
                    ids=[record["id"]],
                    documents=[record["content"]],
                    metadatas=[record["metadata"]],
                    embeddings=[record["embedding"]]
                )
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self.search_with_filter(query, top_k=top_k, metadata_filter=None)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection is not None:
            try:
                query_emb = self._embedding_fn(query)
                chroma_results = self._collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where=metadata_filter,
                    include=["documents", "metadatas", "embeddings"]
                )
                formatted_results = []
                if chroma_results and chroma_results.get("ids") and len(chroma_results["ids"]) > 0:
                    ids = chroma_results["ids"][0]
                    documents = chroma_results["documents"][0]
                    metadatas = chroma_results["metadatas"][0]
                    embeddings = chroma_results["embeddings"][0]
                    for i in range(len(ids)):
                        score = compute_similarity(query_emb, embeddings[i])
                        formatted_results.append({
                            "id": ids[i],
                            "content": documents[i],
                            "metadata": metadatas[i],
                            "score": score
                        })
                    formatted_results.sort(key=lambda x: x["score"], reverse=True)
                    return formatted_results
            except Exception:
                pass

        # In-memory pre-filtering
        filtered_records = []
        for r in self._store:
            matches = True
            if metadata_filter:
                for k, v in metadata_filter.items():
                    if r["metadata"].get(k) != v:
                        matches = False
                        break
            if matches:
                filtered_records.append(r)

        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        deleted_any = False
        if self._use_chroma and self._collection is not None:
            try:
                existing = self._collection.get(where={"doc_id": doc_id})
                if existing and existing.get("ids"):
                    self._collection.delete(where={"doc_id": doc_id})
                    deleted_any = True
            except Exception:
                pass

        initial_len = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        if not self._use_chroma:
            deleted_any = len(self._store) < initial_len
        else:
            # If Chroma is used, check if we also deleted in-memory or chroma succeeded
            deleted_any = deleted_any or (len(self._store) < initial_len)

        return deleted_any
