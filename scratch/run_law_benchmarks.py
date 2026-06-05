import os
from pathlib import Path
from src import Document, EmbeddingStore, KnowledgeBaseAgent, _mock_embed

# Load the 8 chapters of Luật AI
docs = []
for i in range(1, 9):
    path = Path("data") / f"luat_ai_chuong{i}.md"
    if not path.exists():
        continue
    content = path.read_text(encoding="utf-8")
    # Gán metadata mẫu cho từng chương
    meta = {
        "department": "legal" if i in [5, 6, 7] else "general",
        "lang": "vi",
        "chapter": str(i)
    }
    docs.append(Document(id=f"luat_ai_chuong{i}", content=content, metadata=meta))

# Initialize store with MockEmbedder
store = EmbeddingStore(collection_name="law_store", embedding_fn=_mock_embed)
store.add_documents(docs)

# Initialize agent with a mock LLM that returns a clean snippet of the retrieved context
def test_llm(prompt: str) -> str:
    if "Context 1:" in prompt:
        context_segment = prompt.split("Context 1:")[1].split("\n")[0][:100]
        return f"[Mock LLM] Trả lời từ ngữ cảnh: {context_segment.strip()}..."
    return "[Mock LLM] Không tìm thấy ngữ cảnh."

agent = KnowledgeBaseAgent(store=store, llm_fn=test_llm)

# 5 Benchmark queries with filters
queries = [
    ("Phạm vi điều chỉnh của Luật Trí tuệ nhân tạo là gì?", {"chapter": "1"}),
    ("Có mấy mức độ phân loại rủi ro của hệ thống trí tuệ nhân tạo?", {"chapter": "2"}),
    ("Hạ tầng trí tuệ nhân tạo quốc gia bao gồm những gì?", {"chapter": "3"}),
    ("Khung đạo đức trí tuệ nhân tạo quốc gia được ban hành dựa trên nguyên tắc nào?", {"chapter": "5"}),
    ("Hình thức xử lý vi phạm trong hoạt động trí tuệ nhân tạo?", {"chapter": "6"})
]

print("Index size (documents):", store.get_collection_size())
print("\n--- BENCHMARK RESULTS ---")
for idx, (q, filter_dict) in enumerate(queries, 1):
    # Dùng hàm search_with_filter
    results = store.search_with_filter(q, top_k=3, metadata_filter=filter_dict)
    print(f"\nQuery {idx}: {q}")
    print(f"Filter: {filter_dict}")
    if results:
        top_res = results[0]
        print(f"Top-1 Chunk Doc ID: {top_res['id']}")
        print(f"Score: {top_res['score']:.5f}")
        print(f"Content Summary: {top_res['content'][:120].strip().replace('\n', ' ')}")
        ans = agent.answer(q, top_k=3)
        print(f"Agent Answer: {ans}")
    else:
        print("No results found.")
