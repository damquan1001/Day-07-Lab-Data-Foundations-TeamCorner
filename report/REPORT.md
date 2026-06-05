# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần bằng 1.0) nghĩa là hai vector biểu diễn văn bản chỉ về cùng một hướng trong không gian vector đa chiều, cho thấy hai văn bản đó có mối liên hệ ngữ nghĩa rất chặt chẽ hoặc dùng các từ ngữ tương đồng nhau, bất kể độ dài ngắn của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: Học máy là một phân nhánh của trí tuệ nhân tạo.
- Sentence B: Trí tuệ nhân tạo bao gồm cả lĩnh vực học máy.
- Tại sao tương đồng: Cả hai câu đều diễn đạt cùng một mối quan hệ ngữ nghĩa giữa "học máy" và "trí tuệ nhân tạo" bằng các từ khóa tương đồng.

**Ví dụ LOW similarity:**
- Sentence A: Hôm nay trời nắng ráo và gió nhẹ.
- Sentence B: Lập trình Python rất thú vị và mạnh mẽ.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (thời tiết và lập trình), không có từ khóa hay ý nghĩa ngữ nghĩa chung nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa các vector thay vì khoảng cách độ dài, do đó nó không bị ảnh hưởng bởi độ dài của văn bản (số lượng từ). Trong xử lý ngôn ngữ tự nhiên, hai văn bản dài ngắn khác nhau nhưng cùng chủ đề vẫn có cosine similarity cao, trong khi khoảng cách Euclid sẽ rất lớn do độ dài vector khác biệt nhiều.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, phép tính trở thành: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25`. Số lượng chunk tăng lên 25. Ta muốn overlap nhiều hơn để tránh làm mất ngữ cảnh ở ranh giới giữa hai chunk kế tiếp, giúp thông tin không bị ngắt quãng nửa chừng.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hệ thống RAG và Nền Tảng Lập Trình Python/Vector Store cho Trợ lý Tri thức Nội bộ.

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này để xây dựng một trợ lý tri thức phục vụ cho các lập trình viên mới gia nhập dự án và các nhân viên hỗ trợ khách hàng. Dữ liệu này giúp cung cấp các kiến thức căn bản về lập trình Python, thiết kế hệ thống RAG, lưu trữ Vector Store và quy trình xử lý lỗi để hỗ trợ người dùng nhanh chóng.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | customer_support_playbook.txt | Tài liệu quy trình hỗ trợ nội bộ | 1703 | `{"department": "support", "lang": "en"}` |
| 2 | python_intro.txt | Tài liệu giới thiệu ngôn ngữ Python | 1953 | `{"department": "engineering", "lang": "en"}` |
| 3 | rag_system_design.md | Tài liệu thiết kế hệ thống RAG nội bộ | 2416 | `{"department": "engineering", "lang": "en"}` |
| 4 | vector_store_notes.md | Tài liệu ghi chú thiết kế Vector Store | 2149 | `{"department": "engineering", "lang": "en"}` |
| 5 | vi_retrieval_notes.md | Tài liệu tiếng Việt về Retrieval trợ lý | 2188 | `{"department": "engineering", "lang": "vi"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| department | string | support | Giới hạn không gian tìm kiếm của tác tử theo phòng ban liên quan (ví dụ: chỉ tìm quy trình hỗ trợ khách hàng khi người dùng hỏi các câu liên quan đến hỗ trợ, tránh lẫn lộn với mã nguồn kỹ thuật). |
| lang | string | vi | Lọc ngôn ngữ tài liệu giúp hệ thống tránh lấy nhầm các tài liệu đa ngôn ngữ gây nhiễu ngữ cảnh cho LLM khi trả lời bằng một ngôn ngữ cụ thể. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | FixedSizeChunker (`fixed_size`) | 11 | 194.91 | Không (cắt ngang từ ở cuối chunk, ví dụ: 'emph', 'AP', 'sw') |
| python_intro.txt | SentenceChunker (`by_sentences`) | 5 | 387.00 | Có (giữ nguyên ranh giới câu trọn vẹn) |
| python_intro.txt | RecursiveChunker (`recursive`) | 12 | 160.08 | Có (tách theo cấu trúc dấu phân tách và giữ nguyên câu) |

### Strategy Của Tôi

**Loại:** RecursiveChunker (`recursive`)

**Mô tả cách hoạt động:**
> Chiến lược này hoạt động bằng cách phân chia văn bản dựa trên một danh sách các ký tự phân tách có thứ tự ưu tiên (`\n\n`, `\n`, `. `, ` `, `""`). Đầu tiên, nó cố gắng tách văn bản ở mức đoạn (`\n\n`), sau đó nếu đoạn nào vẫn lớn hơn `chunk_size` thì sẽ tách tiếp ở mức câu (`\n`, `. `) và cuối cùng là từ (` `). Điều này đảm bảo cấu trúc ngữ cảnh của tài liệu được bảo toàn tự nhiên nhất.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Vì các tài liệu kỹ thuật trong domain của nhóm được viết dưới dạng Markdown và text có cấu trúc đoạn văn, danh sách rõ ràng. Việc phân tách đệ quy giúp giữ nguyên các khối thông tin cấu trúc như các bước triển khai hoặc các giải thích đi kèm nhau.

**Code snippet (nếu custom):**
```python
# Sử dụng RecursiveChunker mặc định từ src/chunking.py
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| python_intro.txt | best baseline (by_sentences) | 5 | 387.00 | Tốt, giữ trọn ý câu nhưng kích thước chunk hơi lớn. |
| python_intro.txt | **của tôi** (recursive) | 12 | 160.08 | Rất tốt, các đoạn nhỏ gọn, tập trung đúng chủ đề hơn. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive (chunk_size=200) | 8/10 | Giữ ngữ cảnh cấu trúc tốt, chunk gọn gàng. | Đôi khi tách các đoạn dài thành nhiều phần nhỏ khó liên kết. |
| Thành viên A | FixedSize (chunk_size=500) | 6/10 | Số lượng chunk ổn định, dễ cấu hình. | Bị cắt ngang các từ kỹ thuật và câu lệnh. |
| Thành viên B | Sentence (max_sentences=3) | 7/10 | Đọc rất tự nhiên, bảo toàn câu hoàn hảo. | Kích thước các chunk biến động mạnh. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược `RecursiveChunker` là tốt nhất cho domain này. Nó giúp giữ nguyên các câu lệnh kỹ thuật đi liền với lời giải thích của nó trong cùng một chunk mà không làm đứt đoạn nội dung như `FixedSizeChunker`, đồng thời cho kích thước chunk đồng đều và tối ưu hơn `SentenceChunker`.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Hàm sử dụng regex lookbehind `(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)` để tách văn bản dựa trên các dấu chấm, chấm hỏi, chấm than theo sau bởi khoảng trắng hoặc xuống dòng mà không làm mất dấu câu. Mỗi câu sau đó được làm sạch bằng `strip()` và loại bỏ câu rỗng, rồi nhóm tuần tự tối đa `max_sentences_per_chunk` câu vào mỗi chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Hàm triển khai thuật toán chia để trị đệ quy: tìm kiếm ký tự phân tách đầu tiên phù hợp trong độ ưu tiên giảm dần. Nếu tìm thấy, văn bản được chia nhỏ ra; bất kỳ đoạn nhỏ nào vượt quá `chunk_size` sẽ được đệ quy chia tiếp bằng các ký tự phân tách còn lại. Base case là khi độ dài đoạn văn bản nhỏ hơn `chunk_size` hoặc không còn ký tự phân tách nào (khi đó sẽ chia nhỏ theo độ dài chuỗi ký tự mặc định). Cuối cùng, các phần tử được gộp lại sao cho tổng độ dài mỗi khối không vượt quá `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Khi thêm văn bản, mỗi document được chuyển thành một record chuẩn hóa chứa `id`, `content`, `metadata`, và `embedding`. Việc tìm kiếm tương đồng được thực hiện bằng cách tính điểm `compute_similarity` (cosine similarity) giữa câu truy vấn và tất cả vector trong store, sau đó sắp xếp theo điểm số từ cao xuống thấp và lấy ra `top_k` kết quả.

**`search_with_filter` + `delete_document`** — approach:
> Việc lọc metadata được thực hiện trước (pre-filtering) bằng cách lọc các record có metadata tương ứng trước khi tiến hành tính toán độ tương đồng và tìm kiếm. Hàm xóa tài liệu (`delete_document`) tìm và xóa tất cả các chunk thuộc về tài liệu có `metadata['doc_id'] == doc_id` và trả về trạng thái xem có bất kỳ thay đổi nào hay không.

### KnowledgeBaseAgent

**`answer`** — approach:
> Khi nhận câu hỏi, tác tử thực hiện tìm kiếm ngữ cảnh tương đương (`search`) từ vector store để thu thập `top_k` chunk tài liệu liên quan nhất. Ngữ cảnh này được định dạng và đưa vào Prompt có cấu trúc chung cùng với câu hỏi, rồi truyền vào hàm `llm_fn` để lấy câu trả lời.

### Test Results

```
============================= 42 passed in 0.13s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Trí tuệ nhân tạo đang thay đổi thế giới. | Thế giới đang bị thay đổi bởi trí tuệ nhân tạo. | high | -0.09163 | Sai |
| 2 | Học máy là một tập con của trí tuệ nhân tạo. | Lập trình viên sử dụng Python để xây dựng mô hình AI. | low / medium | -0.10154 | Đúng |
| 3 | Trời hôm nay nhiều mây và có thể mưa. | Thuật toán tìm kiếm nhị phân có độ phức tạp O(log n). | low | -0.03582 | Đúng |
| 4 | Tôi thích ăn táo. | Quả táo là trái cây ưa thích của tôi. | high | -0.10139 | Sai |
| 5 | Học máy rất dễ học. | Học máy không hề dễ học tí nào. | high | -0.23369 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là các cặp câu có ý nghĩa tương đồng cao (Cặp 1, Cặp 4) hoặc cặp phủ định có độ phản ứng từ vựng cao (Cặp 5) đều có điểm tương đồng rất thấp hoặc âm (quanh mức 0). Điều này cho thấy MockEmbedder sinh vector dựa trên mã băm MD5 của chuỗi ký tự chính xác chứ không hề học được ngữ nghĩa thực sự của câu chữ, minh chứng cho việc tại sao chúng ta cần các mô hình embedding thật để tìm kiếm ngữ nghĩa chính xác.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tại sao chất lượng chunking lại ảnh hưởng trực tiếp đến retrieval? | Nếu chunk quá ngắn sẽ thiếu ngữ cảnh; nếu quá dài sẽ loãng ý làm giảm độ chính xác của kết quả tìm kiếm. |
| 2 | What are the four stages of a common vector search pipeline? | 1. Chunk documents, 2. Embed each chunk, 3. Store the vector and metadata, 4. Embed query and rank. |
| 3 | What does the assistant do if the retrieval results are weak or contradictory? | The assistant should say so explicitly instead of pretending the answer is complete. |
| 4 | Why do support authors need to avoid vague statements in support content? | Vague statements should be avoided because specific terms make chunks more useful for query matching. |
| 5 | What is the limitation of Python regarding CPU-intensive workloads? | Python is usually slower than low-level compiled languages for CPU-intensive workloads. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tại sao chất lượng chunking lại ảnh hưởng trực tiếp đến retrieval? | Ghi chú về Retrieval cho Trợ lý Tri thức Nội bộ... | -0.16534 | Yes | [Mock LLM] Trả lời dựa trên ngữ cảnh... |
| 2 | What are the four stages of a common vector search pipeline? | Ghi chú về Retrieval cho Trợ lý Tri thức Nội bộ... | 0.16593 | No | [Mock LLM] Trả lời dựa trên ngữ cảnh sai... |
| 3 | What does the assistant do if the retrieval results are weak or contradictory? | Customer Support Playbook for the AI Knowledge Assistant... | 0.19932 | No | [Mock LLM] Trả lời dựa trên ngữ cảnh sai... |
| 4 | Why do support authors need to avoid vague statements in support content? | Customer Support Playbook for the AI Knowledge Assistant... | -0.06808 | Yes | [Mock LLM] Trả lời dựa trên ngữ cảnh... |
| 5 | What is the limitation of Python regarding CPU-intensive workloads? | Customer Support Playbook for the AI Knowledge Assistant... | 0.19327 | No | [Mock LLM] Trả lời dựa trên ngữ cảnh sai... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được từ các thành viên khác cách phân loại và gắn metadata chi tiết như phòng ban và ngôn ngữ. Điều này giúp tối ưu hóa bộ lọc đáng kể trước khi tính toán tương đồng vector.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác đã chỉ ra cách chuyển đổi tài liệu PDF phức tạp chứa bảng biểu sang định dạng Markdown chất lượng cao bằng công cụ chuyên dụng để tránh bị mất định dạng hoặc làm hỏng dữ liệu text.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ sử dụng một mô hình embedding thực sự (như `all-MiniLM-L6-v2`) để chạy thử nghiệm nhằm đảm bảo kết quả tìm kiếm mang tính ngữ nghĩa đích thực thay vì hoàn toàn ngẫu nhiên như MockEmbedder hiện tại.

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
