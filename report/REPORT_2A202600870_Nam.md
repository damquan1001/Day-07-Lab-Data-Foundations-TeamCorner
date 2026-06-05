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

**Domain:** Văn bản pháp luật (Dự thảo Luật Trí tuệ nhân tạo Việt Nam)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì các văn bản luật có tính cấu trúc rất cao (chia theo Chương, Điều, Khoản) và từ khóa dễ bị lặp lại giữa các quy định. Đặc thù này khiến nó trở thành bộ dữ liệu tuyệt vời để thử nghiệm sự khác biệt giữa các chiến lược Chunking (cắt theo ký tự vs cắt theo Điều luật) và làm nổi bật sức mạnh của kỹ thuật Metadata Filtering (ví dụ lọc theo Chương hoặc Mức độ rủi ro) thay vì chỉ phụ thuộc vào Semantic Search thông thường.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `luat_ai_chuong1.md` | Dự thảo Luật AI 2025 | ~10.600 | `{"category": "luat_ai", "chapter": 1}` |
| 2 | `luat_ai_chuong2.md` | Dự thảo Luật AI 2025 | ~13.000 | `{"category": "luat_ai", "chapter": 2}` |
| 3 | `luat_ai_chuong3.md` | Dự thảo Luật AI 2025 | ~7.200 | `{"category": "luat_ai", "chapter": 3}` |
| 4 | `luat_ai_chuong4.md` | Dự thảo Luật AI 2025 | ~14.900 | `{"category": "luat_ai", "chapter": 4}` |
| 5 | `luat_ai_chuong5.md` | Dự thảo Luật AI 2025 | ~3.700 | `{"category": "luat_ai", "chapter": 5}` |
*(Lưu ý: Còn thêm Chương 6, 7, 8 trong thư mục data nhưng nhóm list 5 file đại diện)*

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "luat_ai" | Giúp phân biệt tài liệu luật với các tài liệu nội bộ, FAQ hay hướng dẫn kỹ thuật khác trong cùng một Vector Store lớn. |
| chapter | int | 2 | Vô cùng hữu ích khi người dùng có truy vấn khoanh vùng chủ đề (vd: "Trong phần phân loại rủi ro..."), giúp lọc đi các kết quả gây nhiễu từ chương khác. |
| rui_ro | string | "cao" | Cho phép filter chính xác quy định áp dụng cho từng đối tượng, vì từ khóa "trách nhiệm" có thể xuất hiện rải rác ở khắp văn bản. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu (chunk_size=300):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| luat_ai_chuong1.md | FixedSizeChunker (`fixed_size`) | 40 | 295.4 | Không tốt (cắt ngang câu) |
| luat_ai_chuong1.md | SentenceChunker (`by_sentences`) | 41 | 258.6 | Có (nhưng tách rời ý) |
| luat_ai_chuong1.md | RecursiveChunker (`recursive`) | 639 | 15.7 | Kém (chunk quá nhỏ) |
| luat_ai_chuong2.md | FixedSizeChunker (`fixed_size`) | 49 | 295.0 | Không tốt |
| luat_ai_chuong2.md | SentenceChunker (`by_sentences`) | 42 | 308.9 | Có |
| luat_ai_chuong2.md | RecursiveChunker (`recursive`) | 639 | 19.4 | Kém |

### Strategy Của Tôi

**Loại:** Custom `ArticleChunker`

**Mô tả cách hoạt động:**
> Dùng Regex (Regular Expression) để chia văn bản mỗi khi gặp chuỗi `### Điều`. Cách này sẽ giúp toàn bộ nội dung của một Điều luật nằm trọn vẹn trong một chunk duy nhất, bất kể nó dài hay ngắn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Vì tài liệu nhóm chọn là "Luật Trí tuệ nhân tạo" được cấu trúc rất rõ ràng theo các Điều (Article). Nếu dùng Fixed Size sẽ vô tình làm đứt đoạn quy định pháp lý. Việc chia theo từng "Điều" giúp giữ nguyên được văn cảnh (context) và ý nghĩa trọn vẹn nhất phục vụ cho RAG.

**Code snippet (nếu custom):**
```python
class ArticleChunker:
    def chunk(self, text: str) -> list[str]:
        import re
        parts = re.split(r'(?=\n### Điều)', "\n" + text)
        return [p.strip() for p in parts if p.strip()]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| luat_ai_chuong1.md | SentenceChunker | 41 | 258.6 | trung bình (mất liên kết ngữ cảnh điều luật) |
| luat_ai_chuong1.md | **ArticleChunker** | 9 | 1181.9 | tốt (giữ trọn vẹn ý nghĩa của 1 điều) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | ArticleChunker | 10 | Giữ trọn vẹn ý nghĩa của 1 điều | Số lượng chunk ít, đoạn dài có thể làm mờ ý nhỏ |
| Khoa | FixedSizeChunker (size=200, overlap=0) | 6 | Code đơn giản, tốc độ chunking rất nhanh | Dễ bị cắt ngang câu hoặc giữa đoạn ý quan trọng |
| Quân | SentenceChunker (max=3) | 7 | Giữ nguyên vẹn cấu trúc từng câu | Có thể thiếu ngữ cảnh liên kết từ các câu trước đó |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Dựa trên kết quả thực tế của nhóm, **ArticleChunker** (cắt theo Điều luật) là strategy tốt nhất cho domain "Luật Trí tuệ nhân tạo". Văn bản luật được cấu trúc thành các Điều, Khoản rất chặt chẽ. Việc dùng FixedSize hay SentenceChunker sẽ vô tình chia cắt các Khoản trong cùng một Điều ra làm nhiều đoạn khác nhau, khiến cho LLM không nắm được toàn bộ quy định khi truy vấn. Khi gom chung toàn bộ 1 Điều vào 1 chunk duy nhất, hệ thống RAG trả lời chính xác và đầy đủ ngữ cảnh pháp lý hơn hẳn.

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
| 1 | Python is a high-level programming language. | Machine learning uses algorithms to learn from data. | low | 0.231 | Yes |
| 2 | A fox is a small omnivorous mammal. | Foxes are small mammals that eat plants and animals. | high | 0.892 | Yes |
| 3 | Dogs are loyal companions. | Vector databases store embeddings. | low | -0.104 | Yes |
| 4 | Jumping is a physical activity. | Exercise improves leg strength. | medium | 0.650 | Yes |
| 5 | Natural language processing handles text understanding. | Computer vision processes images. | low | 0.150 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Cặp số (4) "Jumping..." và "Exercise..." có điểm số khá cao (0.650) dù 2 câu không có từ vựng nào chung. Điều này cho thấy Embeddings có khả năng hiểu được "ngữ nghĩa" (semantic) ẩn bên trong chứ không chỉ là khớp từ khóa đơn thuần. Cả 2 câu đều nói về hoạt động thể dục thể chất.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "Sự cố nghiêm trọng" trong hoạt động của hệ thống AI được định nghĩa như thế nào? | Là sự kiện gây ra hoặc có nguy cơ gây thiệt hại đáng kể đến tính mạng, sức khỏe, quyền con người, tài sản... (Điều 3, Khoản 8). |
| 2 | Hệ thống AI ứng dụng trong lĩnh vực y tế thì cần phải quản lý những rủi ro nào? | Phải bảo đảm an toàn cho người bệnh; độ tin cậy trong điều kiện sử dụng thực tế; bảo vệ dữ liệu về sức khỏe (Điều 6, Khoản 2a). |
| 3 | Các hành vi nào bị nghiêm cấm khi sử dụng AI để tạo ra nội dung deepfake? | Cấm sử dụng yếu tố giả mạo để lừa dối có chủ đích, gây tổn hại quyền lợi con người, an ninh quốc gia (Điều 7, Khoản 2b, 2d). |
| 4 | (Cần filter metadata `rui_ro: cao`) Nhà cung cấp hệ thống AI rủi ro cao phải làm gì để minh bạch thông tin? | Thiết kế hệ thống nhận biết tương tác, đánh dấu định dạng máy đọc, và giải trình chức năng (Điều 11 & 14). |
| 5 | (Cần filter metadata `rui_ro: thap`) Chỉ tính riêng trong quy định về rủi ro thấp, người dùng có trách nhiệm gì? | Có quyền sử dụng cho mục đích hợp pháp và tự chịu trách nhiệm trước pháp luật về hoạt động của mình (Điều 15, Khoản 2c). |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | "Sự cố nghiêm trọng"... | Giải thích từ ngữ, Khoản 8: Sự cố nghiêm trọng là sự kiện... | 0.92 | Yes | Là sự kiện gây thiệt hại đáng kể đến tính mạng, sức khỏe, tài sản... |
| 2 | Hệ thống AI y tế... | Ứng dụng AI, Khoản 2a: Lĩnh vực y tế bảo đảm an toàn... | 0.88 | Yes | Phải bảo đảm an toàn cho người bệnh, độ tin cậy và bảo vệ dữ liệu sức khỏe. |
| 3 | Các hành vi bị cấm... | Điều 7, Khoản 2: Sử dụng yếu tố giả mạo để lừa dối... | 0.95 | Yes | Cấm giả mạo người thật để lừa dối, gây tổn hại quyền con người và an ninh quốc gia. |
| 4 | (Metadata cao)... | Điều 11 & 14: Trách nhiệm minh bạch, thông báo... | 0.91 | Yes | Phải thiết kế để người dùng nhận biết được tương tác, đánh dấu định dạng máy đọc. |
| 5 | (Metadata thấp)... | Điều 15, Khoản 2: Hệ thống rủi ro thấp... | 0.87 | Yes | Người dùng được quyền sử dụng hợp pháp và tự chịu trách nhiệm. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách thiết kế Metadata lọc theo cấu trúc chương/điều thay vì phân loại chủ đề chung chung. Việc chia nhỏ các tag metadata giúp hàm filter hoạt động chính xác hơn, tránh bị nhiễu do từ khóa lặp lại giữa các chương.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm bạn dùng cách thiết kế Chunking theo cấu trúc Hỏi-Đáp (Q&A pairs) đối với tài liệu FAQ. Cách này làm tăng đáng kể độ chính xác của thuật toán Cosine Similarity so với việc chia đoạn cố định (Fixed size) thông thường.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ đầu tư nhiều thời gian hơn vào bước làm sạch dữ liệu (Data Cleaning) và tạo ra nhiều lớp Metadata linh hoạt hơn. Đồng thời, tôi sẽ thử nghiệm thêm các kỹ thuật chunking dựa trên Semantic thay vì chỉ dùng regex đơn thuần.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
