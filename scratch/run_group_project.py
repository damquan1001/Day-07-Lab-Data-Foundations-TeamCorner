import os
from pathlib import Path
from src.chunking import RecursiveChunker
from src.models import Document
from src.store import EmbeddingStore
from src.embeddings import _mock_embed, LocalEmbedder
from src.agent import KnowledgeBaseAgent

# 5 Câu hỏi benchmark đã thống nhất của nhóm
BENCHMARK_QUERIES = [
    ("Phạm vi điều chỉnh của Luật Trí tuệ nhân tạo là gì?", {"chapter": "1"}),
    ("Có mấy mức độ phân loại rủi ro của hệ thống trí tuệ nhân tạo?", {"chapter": "2"}),
    ("Hạ tầng trí tuệ nhân tạo quốc gia bao gồm những gì?", {"chapter": "3"}),
    ("Khung đạo đức trí tuệ nhân tạo quốc gia được ban hành dựa trên nguyên tắc nào?", {"chapter": "5"}),
    ("Hình thức xử lý vi phạm trong hoạt động trí tuệ nhân tạo?", {"chapter": "6"})
]

def load_and_chunk_data(chunk_size=400):
    chunker = RecursiveChunker(chunk_size=chunk_size)
    all_documents = []
    
    for i in range(1, 9):
        file_path = Path("data") / f"luat_ai_chuong{i}.md"
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        chunks = chunker.chunk(content)
        
        for idx, chunk_content in enumerate(chunks):
            department = "legal" if i in [5, 6, 7] else "general"
            metadata = {
                "department": department,
                "lang": "vi",
                "chapter": str(i)
            }
            doc = Document(
                id=f"luat_ai_chuong{i}_chunk_{idx}",
                content=chunk_content,
                metadata=metadata
            )
            all_documents.append(doc)
    return all_documents

def mock_llm(prompt: str) -> str:
    """Mock LLM trích xuất câu đầu tiên của context làm câu trả lời để dễ kiểm tra."""
    if "Context 1:" in prompt:
        parts = prompt.split("Context 1:")
        if len(parts) > 1:
            context_line = parts[1].split("\n")[0].strip()
            # Trích xuất 100 ký tự đầu của context để hiển thị
            return f"[Agent Answer] Trả lời: {context_line[:120]}..."
    return "[Agent Answer] Không tìm thấy ngữ cảnh phù hợp để trả lời."

def run_benchmark_for_embedder(embedder_name, embedder_fn, docs):
    print(f"\n=======================================================")
    print(f"CHẠY BENCHMARK VỚI EMBEDDER: {embedder_name.upper()}")
    print(f"=======================================================")
    
    # Khởi tạo Vector Store
    store = EmbeddingStore(collection_name=f"law_store_{embedder_name}", embedding_fn=embedder_fn)
    store.add_documents(docs)
    
    # Khởi tạo Agent
    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)
    
    results_summary = []
    
    for idx, (query, filter_dict) in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\nCâu hỏi {idx}: '{query}'")
        print(f"  Bộ lọc: {filter_dict}")
        
        # Tìm kiếm trong vector store
        search_results = store.search_with_filter(query, top_k=3, metadata_filter=filter_dict)
        
        if search_results:
            top1 = search_results[0]
            print(f"  -> Top-1 Chunk ID: {top1['id']} (Score: {top1['score']:.5f})")
            print(f"  -> Nội dung chunk: {top1['content'][:150].strip().replace('\n', ' ')}...")
            
            # Agent trả lời câu hỏi
            # Lưu ý: Hàm agent.answer sử dụng `search` chứ không dùng `search_with_filter` 
            # theo định nghĩa trong src/agent.py, nhưng để khớp ngữ cảnh chính xác 
            # chúng ta có thể xem cách Agent trả lời câu hỏi không có filter:
            agent_ans = agent.answer(query, top_k=3)
            print(f"  -> Agent trả lời: {agent_ans}")
            
            results_summary.append({
                "query_idx": idx,
                "top1_id": top1['id'],
                "score": top1['score'],
                "content": top1['content'][:150],
                "agent_answer": agent_ans
            })
        else:
            print("  [LỖI] Không tìm thấy kết quả tìm kiếm nào.")
            results_summary.append({
                "query_idx": idx,
                "top1_id": "N/A",
                "score": 0.0,
                "content": "N/A",
                "agent_answer": "N/A"
            })
            
    return results_summary

def main():
    # Bước 1: Phân mảnh tài liệu
    docs = load_and_chunk_data()
    print(f"Đã chuẩn bị xong {len(docs)} chunks.")
    
    # 1. Chạy với MockEmbedder
    mock_results = run_benchmark_for_embedder("mock", _mock_embed, docs)
    
    # 2. Chạy với LocalEmbedder (Real Model)
    print("\nKhởi tạo LocalEmbedder (Mô hình all-MiniLM-L6-v2)...")
    try:
        real_embedder = LocalEmbedder()
        real_results = run_benchmark_for_embedder("real_local", real_embedder, docs)
    except Exception as e:
        print(f"\n[LỖI] Không thể khởi tạo LocalEmbedder: {e}")

if __name__ == "__main__":
    main()
