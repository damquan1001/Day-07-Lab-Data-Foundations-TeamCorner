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

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.store.search(question, top_k=top_k)
        context_blocks = []

        for index, item in enumerate(retrieved, start=1):
            content = item.get("content", "")
            source = item.get("metadata", {}).get("doc_id", "unknown")
            context_blocks.append(f"[{index}] source={source}\n{content}")

        prompt = (
            "Answer the question using only the retrieved context below. "
            "If the answer is not contained in the context, say that you cannot answer.\n\n"
            "Context:\n"
            f"{'\n\n'.join(context_blocks)}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return self.llm_fn(prompt)
