from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


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
            import chromadb  # noqa: F401

            self._use_chroma = False
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        record_id = f"{doc.id}-{self._next_index}"
        self._next_index += 1
        metadata = dict(doc.metadata)
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": record_id,
            "doc_id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
            "tokens": _tokenize(doc.content),
            "title_tokens": _tokenize(str(metadata.get("article_title", ""))),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        results = []
        for record in records:
            results.append(
                {
                    "id": record["id"],
                    "doc_id": record["doc_id"],
                    "content": record["content"],
                    "metadata": dict(record["metadata"]),
                    "score": _dot(query_embedding, record["embedding"]),
                }
            )
        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]

    def _bm25_score_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens or not records:
            return []

        document_count = len(records)
        document_lengths = [len(record["tokens"]) for record in records]
        average_document_length = sum(document_lengths) / document_count if document_count else 0
        if average_document_length == 0:
            return []

        document_frequency: Counter[str] = Counter()
        for record in records:
            document_frequency.update(set(record["tokens"]))

        k1 = 1.5
        b = 0.75
        results = []
        for record in records:
            tokens = record["tokens"]
            token_counts = Counter(tokens)
            document_length = len(tokens)
            score = 0.0

            for token in query_tokens:
                term_frequency = token_counts.get(token, 0)
                if term_frequency == 0:
                    continue

                matches = document_frequency[token]
                inverse_document_frequency = math.log(
                    1 + (document_count - matches + 0.5) / (matches + 0.5)
                )
                denominator = term_frequency + k1 * (
                    1 - b + b * document_length / average_document_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (k1 + 1) / denominator
                )

            title_matches = set(query_tokens) & set(record.get("title_tokens", []))
            score += 2.0 * len(title_matches)

            results.append(
                {
                    "id": record["id"],
                    "doc_id": record["doc_id"],
                    "content": record["content"],
                    "metadata": dict(record["metadata"]),
                    "score": score,
                }
            )

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        filtered_records = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k documents with BM25 keyword relevance."""
        return self._bm25_score_records(query, self._store, top_k)

    def search_bm25_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Run BM25 search after optional exact-match metadata filtering."""
        if not metadata_filter:
            return self.search_bm25(query, top_k=top_k)

        filtered_records = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._bm25_score_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        size_before = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        return len(self._store) < size_before
