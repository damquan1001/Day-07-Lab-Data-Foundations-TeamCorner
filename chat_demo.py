from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import main
from src.embeddings import _mock_embed
from src.store import EmbeddingStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).parent
STATIC_DIR = ROOT_DIR / "static"
LAW_METADATA_FILTER = {"document_type": "law", "language": "vi"}
SUPPORTED_STRATEGIES = {"vector", "bm25", "hybrid"}


def _law_file_path() -> Path:
    configured = ROOT_DIR / main.SAMPLE_FILES[0]
    if configured.exists():
        return configured

    for path in (ROOT_DIR / "data").glob("*.md"):
        if "Trí tuệ nhân tạo" in path.name:
            return path

    raise FileNotFoundError("Could not find the AI law markdown file in data/.")


def get_preset_questions() -> list[str]:
    return list(main.LAW_BENCHMARK_QUERIES)


def build_demo_store() -> tuple[EmbeddingStore, int]:
    docs = main.load_documents_from_files([str(_law_file_path())])
    store = EmbeddingStore(collection_name="chat_demo", embedding_fn=_mock_embed)
    store.add_documents(docs)
    return store, len(docs)


STORE, DOCUMENT_COUNT = build_demo_store()


def _clean_preview(content: str, limit: int = 360) -> str:
    text = " ".join(content.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _result_payload(result: dict) -> dict:
    metadata = result.get("metadata", {})
    payload = {
        "article": metadata.get("article", ""),
        "article_number": metadata.get("article_number", ""),
        "title": metadata.get("article_title", ""),
        "chapter": metadata.get("chapter", ""),
        "score": round(float(result.get("score", 0.0)), 4),
        "preview": _clean_preview(result.get("content", "")),
    }
    if "bm25_score" in result:
        payload["bm25_score"] = round(float(result.get("bm25_score", 0.0)), 4)
    if "vector_score" in result:
        payload["vector_score"] = round(float(result.get("vector_score", 0.0)), 4)
    return payload


def _simple_answer(question: str, results: list[dict], strategy: str) -> str:
    if not results:
        return "Không tìm thấy ngữ cảnh phù hợp trong dữ liệu luật để hỗ trợ câu trả lời."

    top = results[0]
    metadata = top.get("metadata", {})
    article = metadata.get("article", "điều liên quan")
    title = metadata.get("article_title", "")
    preview = _clean_preview(top.get("content", ""), limit=520)
    heading = f"{article} - {title}" if title else article
    return (
        f"Với chiến lược {strategy}, kết quả phù hợp nhất là {heading}. "
        f"Nội dung liên quan: {preview}"
    )


def search_demo(question: str, strategy: str = "hybrid", top_k: int = 3) -> dict:
    question = question.strip()
    strategy = strategy.lower().strip()
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if not question:
        raise ValueError("Question must not be empty.")

    top_k = max(1, min(int(top_k), 10))
    if strategy == "vector":
        raw_results = STORE.search_with_filter(question, top_k=top_k, metadata_filter=LAW_METADATA_FILTER)
    elif strategy == "bm25":
        raw_results = STORE.search_bm25_with_filter(question, top_k=top_k, metadata_filter=LAW_METADATA_FILTER)
    else:
        raw_results = STORE.search_hybrid_with_filter(question, top_k=top_k, metadata_filter=LAW_METADATA_FILTER)

    return {
        "question": question,
        "strategy": strategy,
        "top_k": top_k,
        "answer": _simple_answer(question, raw_results, strategy),
        "results": [_result_payload(result) for result in raw_results],
    }


def compare_demo(question: str, top_k: int = 3) -> dict:
    return {
        "question": question.strip(),
        "top_k": max(1, min(int(top_k), 10)),
        "comparisons": [
            search_demo(question, strategy=strategy, top_k=top_k)
            for strategy in ("vector", "bm25", "hybrid")
        ],
    }


class ChatDemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/questions":
            self._send_json(
                {
                    "questions": get_preset_questions(),
                    "document_count": DOCUMENT_COUNT,
                    "strategies": sorted(SUPPORTED_STRATEGIES),
                }
            )
            return

        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "chat_demo.html", "text/html; charset=utf-8")
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            question = str(payload.get("question", ""))
            top_k = int(payload.get("top_k", 3))

            if parsed.path == "/api/search":
                strategy = str(payload.get("strategy", "hybrid"))
                self._send_json(search_demo(question, strategy=strategy, top_k=top_k))
                return

            if parsed.path == "/api/compare":
                self._send_json(compare_demo(question, top_k=top_k))
                return

            self.send_error(404, "Not found")
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[chat-demo] {self.address_string()} - {format % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), ChatDemoHandler)
    print(f"Chat demo running at http://{host}:{port}")
    print(f"Loaded {DOCUMENT_COUNT} article-level law documents")
    server.serve_forever()


if __name__ == "__main__":
    run()
