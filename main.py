from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SAMPLE_FILES = [
    "data/Luật Trí tuệ nhân tạo.md",
]

LAW_BENCHMARK_QUERIES = [
    "Luật Trí tuệ nhân tạo có hiệu lực từ ngày nào?",
    "Những hành vi nào bị nghiêm cấm khi sử dụng hệ thống trí tuệ nhân tạo?",
    "Khung đạo đức trí tuệ nhân tạo quốc gia dựa trên những nguyên tắc nào?",
    "Cơ quan nào là đầu mối quản lý nhà nước về trí tuệ nhân tạo?",
    "Điều khoản chuyển tiếp quy định thời hạn tuân thủ nào cho hệ thống trí tuệ nhân tạo đã hoạt động trước ngày luật có hiệu lực?",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.extend(_documents_from_file(path, content))

    return documents


def _documents_from_file(path: Path, content: str) -> list[Document]:
    if path.name == "Luật Trí tuệ nhân tạo.md":
        return split_ai_law_into_articles(path, content)

    return [
        Document(
            id=path.stem,
            content=content,
            metadata={"source": str(path), "extension": path.suffix.lower()},
        )
    ]


def _clean_markdown(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_ai_law_into_articles(path: Path, content: str) -> list[Document]:
    """Split the AI law markdown into article-level documents with metadata."""
    documents: list[Document] = []
    current_chapter = ""
    current_article_number = ""
    current_article_title = ""
    current_lines: list[str] = []

    def flush_article() -> None:
        if not current_article_number or not current_lines:
            return

        article_label = f"Điều {current_article_number}"
        article_content = _clean_markdown("\n".join(current_lines))
        documents.append(
            Document(
                id=f"{path.stem}-dieu-{current_article_number}",
                content=article_content,
                metadata={
                    "source": str(path),
                    "extension": path.suffix.lower(),
                    "document_type": "law",
                    "domain": "ai_policy",
                    "language": "vi",
                    "jurisdiction": "vietnam",
                    "chapter": current_chapter,
                    "article": article_label,
                    "article_number": current_article_number,
                    "article_title": current_article_title,
                },
            )
        )

    for line in content.splitlines():
        chapter_match = re.match(r"^##\s+(Chương\s+[IVXLCDM]+:.+)$", line.strip(), flags=re.IGNORECASE)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
            continue

        article_match = re.match(r"^###\s+Điều\s+(\d+)\.\s+(.+)$", line.strip(), flags=re.IGNORECASE)
        if article_match:
            flush_article()
            current_article_number = article_match.group(1)
            current_article_title = article_match.group(2).strip()
            current_lines = [line.strip()]
            continue

        if current_article_number:
            current_lines.append(line)

    flush_article()

    if documents:
        return documents

    return [
        Document(
            id=path.stem,
            content=_clean_markdown(content),
            metadata={
                "source": str(path),
                "extension": path.suffix.lower(),
                "document_type": "law",
                "domain": "ai_policy",
                "language": "vi",
                "jurisdiction": "vietnam",
            },
        )
    ]


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or LAW_BENCHMARK_QUERIES[0]

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} article-level documents in EmbeddingStore")
    print("\n=== Vector Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} article={result['metadata'].get('article')}")
        print(f"   title: {result['metadata'].get('article_title')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== BM25 + Metadata Filter Search Test ===")
    bm25_results = store.search_bm25_with_filter(
        query,
        top_k=3,
        metadata_filter={"document_type": "law", "language": "vi"},
    )
    for index, result in enumerate(bm25_results, start=1):
        print(f"{index}. score={result['score']:.3f} article={result['metadata'].get('article')}")
        print(f"   title: {result['metadata'].get('article_title')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== Hybrid BM25 + Vector Search Test ===")
    hybrid_results = store.search_hybrid_with_filter(
        query,
        top_k=3,
        metadata_filter={"document_type": "law", "language": "vi"},
    )
    for index, result in enumerate(hybrid_results, start=1):
        print(
            f"{index}. score={result['score']:.3f} "
            f"bm25={result['bm25_score']:.3f} vector={result['vector_score']:.3f} "
            f"article={result['metadata'].get('article')}"
        )
        print(f"   title: {result['metadata'].get('article_title')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== Benchmark Queries ===")
    for index, benchmark_query in enumerate(LAW_BENCHMARK_QUERIES, start=1):
        top_result = store.search_hybrid_with_filter(
            benchmark_query,
            top_k=1,
            metadata_filter={"document_type": "law", "language": "vi"},
        )
        article = top_result[0]["metadata"].get("article") if top_result else "no result"
        title = top_result[0]["metadata"].get("article_title") if top_result else ""
        print(f"{index}. {benchmark_query}")
        print(f"   top hybrid article: {article} - {title}")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm, retrieval_strategy="hybrid")
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3, metadata_filter={"document_type": "law", "language": "vi"}))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
