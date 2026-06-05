from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        llm_fn: Callable[[str], str],
        retrieval_strategy: str = "vector",
    ) -> None:
        self.store = store
        self.llm_fn = llm_fn
        self.retrieval_strategy = retrieval_strategy

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if self.retrieval_strategy == "bm25":
            results = self.store.search_bm25_with_filter(
                question,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        elif metadata_filter:
            results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.store.search(question, top_k=top_k)

        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("article") or metadata.get("source") or metadata.get("doc_id") or result.get("doc_id") or "unknown"
            context_blocks.append(
                f"[Context {index} | source={source} | score={result.get('score', 0):.3f}]\n"
                f"{result.get('content', '')}"
            )

        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context was retrieved."
        prompt = (
            "Answer the question using only the retrieved context below. "
            "If the context does not contain enough information, say that the answer is not supported.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
