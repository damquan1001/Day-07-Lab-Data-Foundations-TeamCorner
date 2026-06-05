# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** TeamCorner
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector embedding hướng gần cùng một hướng trong không gian nhiều chiều, tức là hai đoạn văn bản có nghĩa hoặc ngữ cảnh tương tự nhau. Giá trị càng gần 1 thì mức độ tương đồng ngữ nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: Python is widely used for machine learning and data analysis.
- Sentence B: Many teams use Python for AI projects and scientific computing.
- Tại sao tương đồng: Cả hai đều nói về Python trong bối cảnh ML/data, chia sẻ chủ đề và từ khóa liên quan.

**Ví dụ LOW similarity:**
- Sentence A: The refund policy allows returns within 30 days.
- Sentence B: Neural networks use backpropagation to update weights.
- Tại sao khác: Một câu về chính sách khách hàng, một câu về deep learning — không cùng domain hay ý nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài (magnitude) của vector. Text embeddings thường được normalize, và hai câu có nghĩa tương tự có thể có vector cùng hướng dù độ lớn khác nhau — cosine phản ánh đúng mức tương đồng ngữ nghĩa hơn Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `= ceil((10000 - 50) / (500 - 50))`
> `= ceil(9950 / 450)`
> `= ceil(22.11...)`
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900/400) = 25 chunks` — nhiều hơn 2 chunks so với overlap=50 vì bước nhảy (step) nhỏ hơn. Overlap nhiều hơn giúp thông tin ở ranh giới giữa hai chunk không bị cắt mất, giữ ngữ cảnh liên tục khi retrieve — đặc biệt hữu ích khi câu trả lời nằm ngay biên giới hai đoạn text.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật Trí tuệ nhân tạo Việt Nam (134/2025/QH15), tách theo 8 chương

**Tại sao nhóm chọn domain này?**
> Văn bản pháp luật có cấu trúc Chương → Điều rõ ràng, phù hợp RAG tra cứu điều khoản. Tách 8 file theo chương giúp gán metadata `chapter`/`topic` trực tiếp từ tên file, thu hẹp không gian tìm kiếm khi dùng `search_with_filter`. Chủ đề AI + luật liên quan môn học và có gold answer cụ thể.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `luat_ai_chuong1.md` | Luật 134/2025/QH15 — Chương I | 10,218 | `chapter=I`, `topic=general`, 8 Điều |
| 2 | `luat_ai_chuong2.md` | Chương II — Quản lý rủi ro | 13,013 | `chapter=II`, `topic=risk_management`, 7 Điều |
| 3 | `luat_ai_chuong3.md` | Chương III — Hạ tầng | 5,421 | `chapter=III`, `topic=infrastructure`, 3 Điều |
| 4 | `luat_ai_chuong4.md` | Chương IV — Hệ sinh thái | 11,170 | `chapter=IV`, `topic=ecosystem`, 7 Điều |
| 5 | `luat_ai_chuong5.md` | Chương V — Đạo đức | 2,779 | `chapter=V`, `topic=ethics`, 2 Điều |
| 6 | `luat_ai_chuong6.md` | Chương VI — Thanh tra | 2,449 | `chapter=VI`, `topic=enforcement`, 2 Điều |
| 7 | `luat_ai_chuong7.md` | Chương VII — Quản lý NN | 2,440 | `chapter=VII`, `topic=state_management`, 3 Điều |
| 8 | `luat_ai_chuong8.md` | Chương VIII — Thi hành | 1,362 | `chapter=VIII`, `topic=effective_date`, 3 Điều |

**Tổng:** 48,852 ký tự, 35 Điều (chunk theo Điều → 35 vectors)

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | str | `data/luat_ai_chuong2.md` | Truy vết file gốc; filter theo chương cụ thể |
| `chapter` | str | `II` | `search_with_filter(chapter="II")` cho query về rủi ro |
| `topic` | str | `risk_management` | Filter theo chủ đề semantic |
| `article` | int | `9` | Đánh giá retrieval có đúng Điều không |
| `language` | str | `vi` | Phân biệt ngôn ngữ khi mở rộng corpus |
| `law_id` | str | `134/2025/QH15` | Nguồn văn bản pháp lý |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên chuong1, chuong2, chuong8:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| chuong1 | FixedSizeChunker | 26 | 393.0 | Không — cắt giữa khoản |
| chuong1 | SentenceChunker | 28 | 364.0 | Một phần |
| chuong1 | RecursiveChunker | 39 | 261.0 | Khá |
| chuong1 | **ArticleChunker** | **8** | **1248.0** | **Có** — 1 Điều/chunk |
| chuong2 | FixedSizeChunker | 33 | 394.3 | Không |
| chuong2 | RecursiveChunker | 46 | 281.9 | Khá |
| chuong2 | **ArticleChunker** | **7** | **1847.7** | **Có** |
| chuong8 | **ArticleChunker** | **3** | **454.0** | **Có** — Điều 34/35 nguyên vẹn |

### Strategy Của Tôi

**Loại:** ArticleChunker + corpus 8 file theo chương

**Mô tả cách hoạt động:**
> Load 8 file `luat_ai_chuong*.md`, gán metadata `chapter`/`topic` từ tên file. Trong mỗi file, tách theo `### Điều X` → 35 chunk. Mỗi chunk = 1 `Document` với metadata đầy đủ trước khi embed.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Luật có đơn vị pháp lý là Điều; tách file theo chương cho phép filter `chapter` thu hẹp search (vd. chỉ VIII cho query hiệu lực). Kết hợp ArticleChunker giữ nguyên khoản/điểm trong một Điều.

**Code snippet (nếu custom):**
```python
# benchmark_luat.py
parts = re.split(r"(?=### Điều \d+)", text)
chunks = [p.strip() for p in parts if "### Điều" in p]
# metadata chapter từ filename: luat_ai_chuong2.md → chapter="II"
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| chuong8 | RecursiveChunker | 4 | ~340 | Thấp — có thể cắt Điều 35 |
| chuong8 | **ArticleChunker + filter chapter** | 3 | 454 | **Cao** — Q4/Q5 top-1 đúng Điều 34/35 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | ArticleChunker + 8 file + chapter filter | 8/10 (Local top-1) | Q2–Q5 đúng với filter; chuong8 chỉ 3 Điều | Q1 (định nghĩa) vẫn trả Điều 1 |
| [Tên] | RecursiveChunker | — | — | *(cập nhật sau)* |
| [Tên] | SentenceChunker | — | — | *(cập nhật sau)* |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> **ArticleChunker + 8 file + filter `chapter`** — Local đạt 4/5 top-1 đúng (Q2–Q5), so với 1 file không filter chỉ 1/5. Tách chương giúp metadata filter có ý nghĩa thực sự; chunk theo Điều giữ gold answer trong một vector.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])[\s\n]+` để tách text thành các câu sau dấu `.`, `!`, `?` (kèm khoảng trắng hoặc xuống dòng). Sau đó gom `max_sentences_per_chunk` câu liên tiếp thành một chunk và `.strip()` kết quả. Edge case: text rỗng hoặc chỉ whitespace trả về `[]`; `max_sentences_per_chunk` được clamp tối thiểu là 1 trong `__init__`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `_split` nhận text và danh sách separators còn lại. Base case: nếu text ≤ `chunk_size` thì trả về chunk đó; nếu hết separators hoặc separator rỗng thì fallback sang `FixedSizeChunker` cắt cứng. Ngược lại, split theo separator đầu tiên, gom các phần nhỏ vào buffer `current`; khi vượt `chunk_size` thì recurse phần đã gom với separators còn lại.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được chuyển thành record dict qua `_make_record`: lưu `id`, `content`, `embedding` (từ `embedding_fn`), và `metadata` (merge metadata gốc + `doc_id`). Records append vào list in-memory `self._store`. `search` embed query rồi gọi `_search_records` tính dot product với từng embedding, sort giảm dần theo score, trả top-k.

**`search_with_filter` + `delete_document`** — approach:
> Filter **trước** khi search: lọc `self._store` theo metadata key-value khớp hoàn toàn, rồi chạy similarity trên subset. `delete_document` rebuild list loại bỏ records có `metadata["doc_id"] == doc_id`, return `True` nếu số lượng giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)` lấy chunks liên quan, nối `content` bằng `\n\n` thành context block. Prompt gồm instruction + context + question + "Answer:", rồi truyền vào `llm_fn`. LLM (mock hoặc thật) sinh câu trả lời dựa trên context đã inject.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.05s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Cùng 5 cặp câu, dự đoán đặt **trước** khi chạy code. So sánh `MockEmbedder` (pytest) vs `LocalEmbedder` (`all-MiniLM-L6-v2`).

### 5a. MockEmbedder (hash-based — dùng trong pytest)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a high-level programming language. | Java is also a popular programming language. | high | 0.0448 | Không |
| 2 | The cat sat on the mat. | Quantum physics describes subatomic particles. | low | 0.1467 | Có |
| 3 | How do I reset my password? | To reset your password, go to Settings and click Forgot Password. | high | 0.0181 | Không |
| 4 | Vector databases store embeddings. | Embeddings are numerical representations of text for similarity search. | high | -0.1464 | Không |
| 5 | Good morning, how are you? | The weather is sunny today. | low | -0.0270 | Có |

**Dự đoán đúng:** 2 / 5

### 5b. LocalEmbedder (`all-MiniLM-L6-v2`)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a high-level programming language. | Java is also a popular programming language. | high | 0.5417 | Có |
| 2 | The cat sat on the mat. | Quantum physics describes subatomic particles. | low | 0.0211 | Có |
| 3 | How do I reset my password? | To reset your password, go to Settings and click Forgot Password. | high | 0.8172 | Có |
| 4 | Vector databases store embeddings. | Embeddings are numerical representations of text for similarity search. | high | 0.5643 | Có |
| 5 | Good morning, how are you? | The weather is sunny today. | low | 0.3395 | Có |

**Dự đoán đúng:** 5 / 5

### So sánh Mock vs Local

| Cặp | Chủ đề | Mock Score | Local Score | Nhận xét |
|-----|--------|------------|-------------|----------|
| 1 | Ngôn ngữ lập trình | 0.04 | 0.54 | Local nhận diện cùng domain |
| 2 | Không liên quan | 0.15 | 0.02 | Cả hai thấp; local tách rõ hơn |
| 3 | Q&A password | 0.02 | 0.82 | Mock fail; local score rất cao |
| 4 | Embeddings | -0.15 | 0.56 | Mock âm; local phản ánh cùng topic |
| 5 | Chào hỏi / thời tiết | -0.03 | 0.34 | Local vẫn thấp hơn ngưỡng high (0.5) |

**Reflection:** Với MockEmbedder, score gần 0 dù hai câu cùng nghĩa — vì vector sinh từ hash MD5, không encode semantics. LocalEmbedder (`sentence-transformers`) train trên ngữ nghĩa nên 5/5 dự đoán đúng. Điều này cho thấy **chất lượng embedding quyết định** liệu cosine similarity có ý nghĩa hay không; pytest dùng mock chỉ để validate logic code, không đánh giá retrieval thực tế.

---

## 6. Results — Cá nhân (10 điểm)

Setup: 8 file `data/luat_ai_chuong*.md` → ArticleChunker (35 chunks) + `search_with_filter(chapter=...)`. Chạy `python benchmark_luat.py`. Kết quả: `report/benchmark_luat_results.json`.

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer | Filter | Chunk |
|---|-------|-------------|--------|-------|
| 1 | Trí tuệ nhân tạo được định nghĩa thế nào? | Học tập, suy luận, nhận thức, phán đoán, hiểu ngôn ngữ tự nhiên (bằng điện tử) | `chapter=I` | Điều 3 |
| 2 | Phân loại mức độ rủi ro? | Cao, trung bình, thấp, không đáng kể | `chapter=II` | Điều 9 |
| 3 | Cơ quan đầu mối quản lý TTNT? | Bộ Khoa học và Công nghệ | `chapter=VII` | Điều 30 |
| 4 | Luật có hiệu lực từ ngày nào? | 01/03/2026 | `chapter=VIII` | Điều 34 |
| 5 | Y tế/giáo dục/tài chính bao lâu tuân thủ? | 18 tháng | `chapter=VIII` | Điều 35 |

### 6a. Kết quả với MockEmbedder (+ chapter filter)

| # | Query | Top-1 Chunk | Score | Top-1 OK? | Top-3 OK? |
|---|-------|-------------|-------|-----------|-----------|
| 1 | Định nghĩa TTNT | Điều 7 (chuong1) | 0.099 | Không | **Có** |
| 2 | Phân loại rủi ro | Điều 15 (chuong2) | 0.085 | Không | **Có** (Điều 9 trong top-3) |
| 3 | Cơ quan đầu mối | Điều 30 (chuong7) | 0.133 | **Có** | Có |
| 4 | Hiệu lực | Điều 34 (chuong8) | 0.171 | **Có** | Có |
| 5 | 18 tháng tuân thủ | Điều 35 (chuong8) | 0.014 | **Có** | Có |

**Relevant trong top-3:** 5 / 5 | **Top-1 đúng:** 3 / 5

### 6b. Kết quả với LocalEmbedder (+ chapter filter)

| # | Query | Top-1 Chunk | Score | Top-1 OK? | Top-3 OK? |
|---|-------|-------------|-------|-----------|-----------|
| 1 | Định nghĩa TTNT | Điều 1 (chuong1) | 0.747 | Không | Không |
| 2 | Phân loại rủi ro | Điều 9 (chuong2) | 0.691 | **Có** | Có |
| 3 | Cơ quan đầu mối | Điều 30 (chuong7) | 0.636 | **Có** | Có |
| 4 | Hiệu lực | Điều 34 (chuong8) | 0.589 | **Có** | Có |
| 5 | 18 tháng tuân thủ | Điều 35 (chuong8) | 0.728 | **Có** | Có |

**Relevant trong top-3:** 4 / 5 | **Top-1 đúng:** 4 / 5

### So sánh: 1 file vs 8 file + filter

| Setup | Mock top-3 | Local top-3 | Local top-1 |
|-------|------------|-------------|-------------|
| 1 file, filter `topic` | 2/5 | 1/5 | 1/5 |
| **8 file, filter `chapter`** | **5/5** | **4/5** | **4/5** |

**Nhận xét:** Tách 8 file + filter `chapter` cải thiện mạnh — đặc biệt Q4 (Điều 34) và Q5 (Điều 35) nhờ `chuong8` chỉ 3 Điều. Q1 vẫn khó: Điều 3 nằm trong Điều 3 chunk lớn (định nghĩa 8 thuật ngữ), Local ưu tiên Điều 1 (phạm vi điều chỉnh). Cần chunk Điều 3 theo khoản hoặc embedder đa ngôn ngữ.

---

## 7. What I Learned (5 điểm — Demo)

**Failure case còn lại — Q1 (định nghĩa TTNT):**
> Dù filter `chapter=I`, Local vẫn trả Điều 1 thay vì Điều 3. Điều 3 chứa 8 định nghĩa trong một chunk ~1.5k ký tự — định nghĩa "trí tuệ nhân tạo" chỉ là khoản 1, dễ bị lu mờ. Mock có Điều 3 trong top-3 nhưng top-1 là Điều 7.

**Điều hay nhất từ data strategy 8 file:**
> Metadata `chapter` gán từ tên file đơn giản và hiệu quả hơn parse từ nội dung. `chuong8` (3 Điều) + filter → Q4/Q5 top-1 đúng 100% với Local.

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *(Cập nhật sau demo — so sánh Recursive vs ArticleChunker.)*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Chunk Điều 3 theo từng khoản định nghĩa (8 sub-chunks). (2) Dùng embedder đa ngôn ngữ cho tiếng Việt. (3) Prefix chunk: `"Điều X — [tên]: [nội dung]"` trước khi embed.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | — / 5 |
| **Tổng (cá nhân hoàn thành)** | | **77 / 85** |
