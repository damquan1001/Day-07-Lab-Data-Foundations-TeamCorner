# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Hoàng Nam
**Nhóm:** TeamCornor
**Ngày:** 05-06-2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity đo độ tương đồng góc giữa hai vectơ embedding. Khi hai câu có ý nghĩa giống nhau, vectơ của chúng gần nhau trong không gian không gian, nên cosine similarity cao gần bằng 1.

**Ví dụ HIGH similarity:**
- Sentence A: "A fox is a small omnivorous mammal."
- Sentence B: "Foxes are small mammals that eat plants and animals."
- Tại sao tương đồng: Hai câu đều nói về đặc điểm và chế độ ăn của cáo.

**Ví dụ LOW similarity:**
- Sentence A: "Python is a high-level programming language."
- Sentence B: "Vector databases store embeddings for similarity search."
- Tại sao khác: Nội dung một câu về lập trình, câu kia về lưu trữ dữ liệu embedding.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Bởi vì cosine similarity chỉ xét hướng của vectơ, không bị ảnh hưởng bởi độ lớn. Text embedding thường được chuẩn hóa và mục tiêu là đo đâu là nghĩa giống nhau thay vì độ lớn chuỗi.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Sử dụng công thức: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`.
> **Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100, số chunk là `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`. Overlap nhiều giúp giữ ngữ cảnh xuyên suốt giữa các chunk, giảm khả năng mất mảnh thông tin quan trọng ở mép chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ]

**Tại sao nhóm chọn domain này?**
> [Nhóm chọn domain này vì nó có nhiều văn bản cấu trúc rõ ràng, phù hợp để so sánh các chiến lược chunking và metadata filtering.]

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "python_intro" | Giúp lọc theo nguồn tài liệu |
| language | string | "vi" | Dùng để tìm câu trả lời phù hợp ngôn ngữ |
| category | string | "productivity" | Hạn chế kết quả xuống domain liên quan |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| [Tên tài liệu] | FixedSizeChunker (`fixed_size`) | [số] | [độ dài] | [Có/Không] |
| [Tên tài liệu] | SentenceChunker (`by_sentences`) | [số] | [độ dài] | [Có/Không] |
| [Tên tài liệu] | RecursiveChunker (`recursive`) | [số] | [độ dài] | [Có/Không] |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> `RecursiveChunker` thử chia theo thứ tự phân tách ưu tiên: đoạn (`\n\n`), dòng (`\n`), câu (`. `), và cuối cùng ký tự trắng. Nếu một đoạn vẫn quá dài, nó sẽ đệ quy xuống separator tiếp theo và chỉ xuống chunk_size khi đã hết separator.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Strategy này phù hợp với tài liệu hỗn hợp vì nó cố gắng giữ các đoạn có ý nghĩa nguyên vẹn trước khi cắt nhỏ. Với nhiều text dạng câu đoạn, nó vừa giữ ngữ cảnh vừa đảm bảo chunk không quá dài.

**Code snippet (nếu custom):**
```python
# RecursiveChunker có sẵn trong src/chunking.py
chunks = RecursiveChunker(chunk_size=200).chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| [Tên tài liệu] | best baseline | [số] | [độ dài] | [tốt/trung bình/kém] |
| [Tên tài liệu] | **của tôi** | [số] | [độ dài] | [tốt/trung bình/kém] |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | [score] | Giữ ngữ cảnh tốt | Có thể tạo nhiều chunk khi text dài |
| [Tên] | [strategy khác] | [score] | | |
| [Tên] | [strategy khác] | [score] | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> [Viết ý kiến dựa trên kết quả thực tế của nhóm.]

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex để tách câu theo dấu chấm, chấm hỏi, chấm than và xuống dòng. Mỗi chunk gom tối đa `max_sentences_per_chunk` câu và giữ lại dấu chấm của câu để nội dung dễ đọc.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy thử split theo priority separators. Nếu một đoạn quá dài, nó dùng separator tiếp theo để chia nhỏ hơn. Base case là khi đoạn đủ ngắn hoặc không còn separator nào nữa, khi đó cắt theo chunk_size.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu từng document cùng embedding và metadata trong store in-memory. Search tạo embedding query rồi dùng dot product để xếp hạng kết quả, trả về top_k item.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` trước lọc metadata rồi mới search trên bản ghi đã lọc. `delete_document` loại bỏ mọi record có cùng `doc_id` từ store.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent lấy top_k chunk từ store, xếp chúng vào prompt dưới dạng context có số thứ tự và nguồn. Nếu không tìm thấy nội dung, prompt yêu cầu trả lời dựa trên context hoặc nói không thể trả lời.

### Test Results

```
42 passed in 0.10s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a high-level programming language. | Machine learning uses algorithms to learn from data. | low | [score] | [Yes/No] |
| 2 | A fox is a small omnivorous mammal. | Foxes are small mammals that eat plants and animals. | high | [score] | [Yes/No] |
| 3 | Dogs are loyal companions. | Vector databases store embeddings. | low | [score] | [Yes/No] |
| 4 | Jumping is a physical activity. | Exercise improves leg strength. | medium | [score] | [Yes/No] |
| 5 | Natural language processing handles text understanding. | Computer vision processes images. | low | [score] | [Yes/No] |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> [Điền sau khi chạy `compute_similarity` với các cặp thực tế.]

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> [Viết 2-3 câu về chiến lược khác hoặc cách họ sử dụng metadata/chunking.]

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> [Viết 2-3 câu về điểm mạnh của cách tiếp cận nhóm khác hoặc bài học rút ra.]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
