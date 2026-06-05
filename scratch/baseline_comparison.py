import pathlib
from src import ChunkingStrategyComparator

def run_compare():
    comparator = ChunkingStrategyComparator()
    
    for ch in ["chuong1", "chuong2"]:
        path = pathlib.Path(f"data/luat_ai_chuong{ch[-1]}.md")
        if not path.exists():
            print(f"File {path} not found.")
            continue
            
        text = path.read_text(encoding="utf-8")
        print(f"\n--- Comparison for {path.name} (length: {len(text)} chars) ---")
        result = comparator.compare(text, chunk_size=300)
        
        for name, stats in result.items():
            print(f"Strategy: {name}")
            print(f"  Chunk Count: {stats['count']}")
            print(f"  Avg Length : {stats['avg_length']:.2f}")

if __name__ == "__main__":
    run_compare()
