import os
from pathlib import Path
from src.chunking import RecursiveChunker
from src.models import Document
from src.store import EmbeddingStore
from src.embeddings import LocalEmbedder

BENCHMARK_QUERIES = [
    ("Phạm vi điều chỉnh của Luật Trí tuệ nhân tạo là gì?", {"chapter": "1"}),
    ("Có mấy mức độ phân loại rủi ro của hệ thống trí tuệ nhân tạo?", {"chapter": "2"}),
    ("Hạ tầng trí tuệ nhân tạo quốc gia bao gồm những gì?", {"chapter": "3"}),
    ("Khung đạo đức trí tuệ nhân tạo quốc gia được ban hành dựa trên nguyên tắc nào?", {"chapter": "5"}),
    ("Hình thức xử lý vi phạm trong hoạt động trí tuệ nhân tạo?", {"chapter": "6"})
]

def main():
    chunker = RecursiveChunker(chunk_size=400)
    all_documents = []
    
    for i in range(1, 9):
        file_path = Path("data") / f"luat_ai_chuong{i}.md"
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        chunks = chunker.chunk(content)
        for idx, chunk_content in enumerate(chunks):
            department = "legal" if i in [5, 6, 7] else "general"
            doc = Document(
                id=f"luat_ai_chuong{i}_chunk_{idx}",
                content=chunk_content,
                metadata={"department": department, "lang": "vi", "chapter": str(i)}
            )
            all_documents.append(doc)
            
    print(f"Loaded {len(all_documents)} chunks.")
    embedder = LocalEmbedder()
    store = EmbeddingStore(collection_name="law_store_top3", embedding_fn=embedder)
    store.add_documents(all_documents)
    
    print("\n--- TOP-3 RESULTS FOR LOCAL EMBEDDER ---")
    for idx, (query, filter_dict) in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\nQuery {idx}: '{query}' (Filter: {filter_dict})")
        results = store.search_with_filter(query, top_k=3, metadata_filter=filter_dict)
        for rank, res in enumerate(results, 1):
            print(f"  Rank {rank}: {res['id']} (Score: {res['score']:.5f})")
            print(f"    Content: {res['content'].strip().replace('\n', ' ')[:180]}...")

if __name__ == "__main__":
    main()
