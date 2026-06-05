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

    def __init__(self, store, llm_fn):
        self.store = store
        self.llm_fn = llm_fn
        
    def answer(self, question, top_k=3):
        results = self.store.search(question, top_k=top_k)
        context = "\n\n".join(r["content"] for r in results)
        prompt = f"""Use the following context to answer the question.
    Context:
    {context}
    Question: {question}
    Answer:"""
        return self.llm_fn(prompt)
