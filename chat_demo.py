from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai

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

load_dotenv(ROOT_DIR / ".env", override=False)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


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


def get_llm_status() -> dict:
    return {
        "provider": "gemini" if GEMINI_API_KEY else "retrieval-preview",
        "model": GEMINI_MODEL if GEMINI_API_KEY else "",
        "available": bool(GEMINI_API_KEY),
    }


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


def _build_grounded_prompt(question: str, results: list[dict], strategy: str) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        article = metadata.get("article", "Điều chưa rõ")
        title = metadata.get("article_title", "")
        chapter = metadata.get("chapter", "")
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] {article} - {title}".strip(),
                    f"Chapter: {chapter}",
                    f"Retrieval score: {float(result.get('score', 0.0)):.4f}",
                    result.get("content", ""),
                ]
            )
        )

    return (
        "Bạn là trợ lý tra cứu Luật Trí tuệ nhân tạo Việt Nam. "
        "Chỉ trả lời dựa trên ngữ cảnh được cung cấp. "
        "Nếu ngữ cảnh không đủ để kết luận, hãy nói rõ là chưa đủ căn cứ. "
        "Trả lời bằng tiếng Việt, ngắn gọn, có nhắc Điều liên quan khi có thể.\n\n"
        f"Chiến lược retrieval: {strategy}\n"
        f"Câu hỏi: {question}\n\n"
        "Ngữ cảnh retrieved:\n"
        f"{'\n\n'.join(context_blocks)}\n\n"
        "Câu trả lời:"
    )


def _gemini_answer(question: str, results: list[dict], strategy: str) -> tuple[str, str]:
    if GEMINI_CLIENT is None:
        return _simple_answer(question, results, strategy), "retrieval-preview"
    if not results:
        return _simple_answer(question, results, strategy), "retrieval-preview"

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_grounded_prompt(question, results, strategy),
            config={
                "temperature": 0.2,
                "max_output_tokens": 512,
            },
        )
        answer = (response.text or "").strip()
        if answer:
            return answer, f"gemini:{GEMINI_MODEL}"
    except Exception as exc:
        print(f"[chat-demo] Gemini fallback: {exc}", file=sys.stderr)

    return _simple_answer(question, results, strategy), "retrieval-preview"


def search_demo(question: str, strategy: str = "hybrid", top_k: int = 3, use_gemini: bool = False) -> dict:
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

    if use_gemini:
        answer, answer_source = _gemini_answer(question, raw_results, strategy)
    else:
        answer = _simple_answer(question, raw_results, strategy)
        answer_source = "retrieval-preview"

    return {
        "question": question,
        "strategy": strategy,
        "top_k": top_k,
        "answer": answer,
        "answer_source": answer_source,
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
                    "llm": get_llm_status(),
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
            use_gemini = bool(payload.get("use_gemini", False))

            if parsed.path == "/api/search":
                strategy = str(payload.get("strategy", "hybrid"))
                self._send_json(search_demo(question, strategy=strategy, top_k=top_k, use_gemini=use_gemini))
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
