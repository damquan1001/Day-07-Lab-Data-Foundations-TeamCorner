"""Benchmark retrieval on Luật TTNT corpus split by chapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.chunking import ChunkingStrategyComparator
from src.embeddings import LocalEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path("data")
CHAPTER_FILES = sorted(DATA_DIR.glob("luat_ai_chuong*.md"))

CHAPTER_META = {
    1: ("I", "general"),
    2: ("II", "risk_management"),
    3: ("III", "infrastructure"),
    4: ("IV", "ecosystem"),
    5: ("V", "ethics"),
    6: ("VI", "enforcement"),
    7: ("VII", "state_management"),
    8: ("VIII", "effective_date"),
}

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Trí tuệ nhân tạo được định nghĩa thế nào trong Luật?",
        "gold": "Thực hiện bằng điện tử các năng lực trí tuệ con người: học tập, suy luận, nhận thức, phán đoán và hiểu ngôn ngữ tự nhiên.",
        "expected_article": 3,
        "filter": {"chapter": "I"},
    },
    {
        "id": 2,
        "query": "Hệ thống trí tuệ nhân tạo được phân loại theo những mức độ rủi ro nào?",
        "gold": "Rủi ro cao, rủi ro trung bình, rủi ro thấp và không đáng kể.",
        "expected_article": 9,
        "filter": {"chapter": "II"},
    },
    {
        "id": 3,
        "query": "Bộ nào là cơ quan đầu mối quản lý trí tuệ nhân tạo?",
        "gold": "Bộ Khoa học và Công nghệ.",
        "expected_article": 30,
        "filter": {"chapter": "VII"},
    },
    {
        "id": 4,
        "query": "Luật Trí tuệ nhân tạo có hiệu lực từ ngày nào?",
        "gold": "01 tháng 3 năm 2026.",
        "expected_article": 34,
        "filter": {"chapter": "VIII"},
    },
    {
        "id": 5,
        "query": "Hệ thống trí tuệ nhân tạo trong lĩnh vực y tế, giáo dục và tài chính có bao lâu để tuân thủ Luật?",
        "gold": "18 tháng kể từ ngày Luật có hiệu lực thi hành.",
        "expected_article": 35,
        "filter": {"chapter": "VIII"},
    },
]


def _chapter_num_from_path(path: Path) -> int:
    match = re.search(r"chuong(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse chapter number from {path.name}")
    return int(match.group(1))


def article_chunk(text: str) -> list[str]:
    parts = re.split(r"(?=### Điều \d+)", text)
    return [part.strip() for part in parts if part.strip() and "### Điều" in part]


def _parse_article_number(chunk: str) -> int | None:
    match = re.search(r"### Điều (\d+)", chunk)
    return int(match.group(1)) if match else None


def build_documents_from_chapters(files: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in files:
        chapter_num = _chapter_num_from_path(path)
        chapter_roman, topic = CHAPTER_META[chapter_num]
        source = str(path).replace("\\", "/")
        text = path.read_text(encoding="utf-8")

        for chunk in article_chunk(text):
            article = _parse_article_number(chunk)
            doc_id = f"chuong{chapter_num}-dieu{article or 'x'}"
            docs.append(
                Document(
                    id=doc_id,
                    content=chunk,
                    metadata={
                        "source": source,
                        "language": "vi",
                        "doc_type": "law",
                        "law_id": "134/2025/QH15",
                        "chapter": chapter_roman,
                        "chapter_num": chapter_num,
                        "article": article,
                        "topic": topic,
                    },
                )
            )
    return docs


def inventory(files: list[Path]) -> list[dict]:
    rows = []
    for path in files:
        chapter_num = _chapter_num_from_path(path)
        chapter_roman, topic = CHAPTER_META[chapter_num]
        text = path.read_text(encoding="utf-8")
        rows.append(
            {
                "file": path.name,
                "source": str(path).replace("\\", "/"),
                "chars": len(text),
                "chapter": chapter_roman,
                "topic": topic,
                "articles": len(article_chunk(text)),
            }
        )
    return rows


def is_relevant(result: dict | None, expected_article: int | None) -> bool:
    if not result or expected_article is None:
        return False
    return result["metadata"].get("article") == expected_article


def summarize_chunk(content: str, limit: int = 80) -> str:
    match = re.search(r"### Điều \d+\.[^\n]*", content)
    if match:
        return match.group(0).replace("### ", "")[:limit]
    return " ".join(content.split())[:limit]


def run_baseline(files: list[Path]) -> dict:
    baseline: dict[str, dict] = {}
    sample_names = {"luat_ai_chuong1.md", "luat_ai_chuong2.md", "luat_ai_chuong8.md"}
    sample_files = [p for p in files if p.name in sample_names]
    for path in sample_files:
        text = path.read_text(encoding="utf-8")
        stats = ChunkingStrategyComparator().compare(text, chunk_size=400)
        articles = article_chunk(text)
        avg_article = sum(len(c) for c in articles) / len(articles) if articles else 0
        baseline[path.name] = {
            "fixed_size": {"count": stats["fixed_size"]["count"], "avg_length": round(stats["fixed_size"]["avg_length"], 1)},
            "by_sentences": {"count": stats["by_sentences"]["count"], "avg_length": round(stats["by_sentences"]["avg_length"], 1)},
            "recursive": {"count": stats["recursive"]["count"], "avg_length": round(stats["recursive"]["avg_length"], 1)},
            "by_article": {"count": len(articles), "avg_length": round(avg_article, 1)},
        }
    return baseline


def run_benchmark(embedder_name: str, embed_fn: Callable[[str], list[float]], docs: list[Document]) -> dict:
    store = EmbeddingStore(collection_name=f"luat_chapters_{embedder_name}", embedding_fn=embed_fn)
    store.add_documents(docs)

    query_results = []
    for item in BENCHMARK_QUERIES:
        filt = item.get("filter")
        results = store.search_with_filter(item["query"], top_k=3, metadata_filter=filt)
        top1 = results[0] if results else None
        query_results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "gold": item["gold"],
                "expected_article": item["expected_article"],
                "filter": filt,
                "top1": {
                    "score": round(top1["score"], 4) if top1 else None,
                    "article": top1["metadata"].get("article") if top1 else None,
                    "chapter": top1["metadata"].get("chapter") if top1 else None,
                    "source": top1["metadata"].get("source") if top1 else None,
                    "summary": summarize_chunk(top1["content"]) if top1 else "",
                    "relevant": is_relevant(top1, item["expected_article"]),
                },
                "top3_relevant": any(is_relevant(r, item["expected_article"]) for r in results[:3]),
            }
        )

    return {
        "embedder": embedder_name,
        "chunk_count": len(docs),
        "relevant_top3": sum(1 for q in query_results if q["top3_relevant"]),
        "queries": query_results,
    }


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> int:
    load_dotenv(override=False)
    if not CHAPTER_FILES:
        print("No chapter files found in data/luat_ai_chuong*.md")
        return 1

    inv = inventory(CHAPTER_FILES)
    docs = build_documents_from_chapters(CHAPTER_FILES)
    print(f"Loaded {len(CHAPTER_FILES)} chapter files -> {len(docs)} article chunks")

    baseline = run_baseline(CHAPTER_FILES)
    print("\n=== Baseline (chuong1, chuong2, chuong8) ===")
    for fname, stats in baseline.items():
        print(f"  {fname}: by_article count={stats['by_article']['count']}, avg={stats['by_article']['avg_length']}")

    output = {"inventory": inv, "baseline": baseline, "total_chunks": len(docs)}

    print("\n=== MockEmbedder (with chapter filter) ===")
    mock_result = run_benchmark("mock", _mock_embed, docs)
    output["mock"] = mock_result
    print(f"Relevant top-3: {mock_result['relevant_top3']} / 5")
    for q in mock_result["queries"]:
        _safe_print(
            f"  Q{q['id']}: art={q['top1']['article']} ch={q['top1']['chapter']} "
            f"score={q['top1']['score']} ok={q['top1']['relevant']} | {q['top1']['summary']}"
        )

    try:
        local_embedder = LocalEmbedder()
        print("\n=== LocalEmbedder (with chapter filter) ===")
        local_result = run_benchmark("local", local_embedder, docs)
        output["local"] = local_result
        print(f"Relevant top-3: {local_result['relevant_top3']} / 5")
        for q in local_result["queries"]:
            _safe_print(
                f"  Q{q['id']}: art={q['top1']['article']} ch={q['top1']['chapter']} "
                f"score={q['top1']['score']} ok={q['top1']['relevant']} | {q['top1']['summary']}"
            )
    except Exception as exc:
        print(f"LocalEmbedder skipped: {exc}")
        output["local_error"] = str(exc)

    out_path = Path("report/benchmark_luat_results.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
