# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Đàm Quân  
**Nhóm:** TeamCorner  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector biểu diễn hai đoạn văn bản đang gần cùng hướng, nên nội dung có khả năng giống nhau về ý nghĩa hoặc ngữ cảnh. Với text embeddings, điểm càng gần `1.0` thì hai câu/chunk càng có xu hướng nói về cùng một chủ đề.

**Ví dụ HIGH similarity:**
- Sentence A: Python is used for machine learning.
- Sentence B: Python is commonly used to train AI models.
- Tại sao tương đồng: Cả hai câu đều nói về Python trong ngữ cảnh machine learning/AI.

**Ví dụ LOW similarity:**
- Sentence A: The AI law defines prohibited behaviors.
- Sentence B: Bananas are yellow fruit.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau, một câu về luật AI và một câu về trái cây.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector hơn là độ lớn tuyệt đối, nên phù hợp khi ta quan tâm văn bản có cùng ý nghĩa hay không. Với embeddings, độ dài vector có thể bị ảnh hưởng bởi cách model biểu diễn, còn hướng vector thường phản ánh quan hệ ngữ nghĩa tốt hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Công thức:

```text
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
num_chunks = ceil((10000 - 50) / (500 - 50))
num_chunks = ceil(9950 / 450)
num_chunks = ceil(22.11)
num_chunks = 23
```

**Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Khi overlap là 100:

```text
num_chunks = ceil((10000 - 100) / (500 - 100))
num_chunks = ceil(9900 / 400)
num_chunks = ceil(24.75)
num_chunks = 25
```

Số chunk tăng từ 23 lên 25 vì mỗi bước trượt ngắn hơn. Overlap nhiều hơn giúp giữ ngữ cảnh giữa các chunk liền kề, nhưng đổi lại làm tăng số chunk cần lưu và tìm kiếm.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật Trí tuệ nhân tạo Việt Nam (Luật 134/2025/QH15), tách theo 8 chương.

**Tại sao nhóm chọn domain này?**  
> Nhóm chọn domain văn bản luật vì cấu trúc Chương -> Điều -> Khoản rất rõ ràng, phù hợp để thử nghiệm chunking theo cấu trúc pháp lý và metadata filtering. Chủ đề AI cũng có nhiều truy vấn thực tế như phân loại rủi ro, hành vi bị cấm, đạo đức AI, hiệu lực thi hành và trách nhiệm quản lý nhà nước.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `luat_ai_chuong1.md` | Luật 134/2025/QH15 — Chương I | 10,218 | `chapter=I`, `topic=general`, `language=vi` |
| 2 | `luat_ai_chuong2.md` | Chương II — Quản lý rủi ro | 13,013 | `chapter=II`, `topic=risk_management`, `language=vi` |
| 3 | `luat_ai_chuong3.md` | Chương III — Hạ tầng | 5,421 | `chapter=III`, `topic=infrastructure`, `language=vi` |
| 4 | `luat_ai_chuong4.md` | Chương IV — Hệ sinh thái | 11,170 | `chapter=IV`, `topic=ecosystem`, `language=vi` |
| 5 | `luat_ai_chuong5.md` | Chương V — Đạo đức | 2,779 | `chapter=V`, `topic=ethics`, `language=vi` |
| 6 | `luat_ai_chuong6.md` | Chương VI — Thanh tra | 2,449 | `chapter=VI`, `topic=enforcement`, `language=vi` |
| 7 | `luat_ai_chuong7.md` | Chương VII — Quản lý nhà nước | 2,440 | `chapter=VII`, `topic=state_management`, `language=vi` |
| 8 | `luat_ai_chuong8.md` | Chương VIII — Thi hành | 1,362 | `chapter=VIII`, `topic=effective_date`, `language=vi` |

**Tổng:** 48,852 ký tự, 35 Điều luật.

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `document_type` | str | `law` | Phân biệt văn bản luật với FAQ, ghi chú hoặc tài liệu demo khác trong cùng store. |
| `language` | str | `vi` | Tránh trộn kết quả khi corpus mở rộng sang tài liệu tiếng Anh hoặc song ngữ. |
| `jurisdiction` | str | `VN` | Gắn phạm vi pháp lý của văn bản, hữu ích nếu sau này thêm luật từ nhiều quốc gia. |
| `chapter` | str | `VIII` | Filter theo chương để giảm nhiễu, đặc biệt với câu hỏi về hiệu lực hoặc chuyển tiếp. |
| `article_number` | int | `34` | Kiểm tra retrieval có đúng Điều mục tiêu hay không. |
| `article_title` | str | `Hiệu lực thi hành` | BM25/hybrid có thể tận dụng match tiêu đề Điều để tăng độ chính xác. |
| `topic` | str | `risk_management` | Thu hẹp truy vấn theo chủ đề như rủi ro, đạo đức, hạ tầng, quản lý nhà nước. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Tôi chạy `ChunkingStrategyComparator().compare()` trên file `data/Luật Trí tuệ nhân tạo.md` với `chunk_size=1000`.

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Luật Trí tuệ nhân tạo | FixedSizeChunker (`fixed_size`) | 50 | 990.9 | Trung bình; dễ cắt ngang điều/khoản |
| Luật Trí tuệ nhân tạo | SentenceChunker (`by_sentences`) | 120 | 411.9 | Tốt ở mức câu, nhưng chia quá nhỏ cho văn bản luật |
| Luật Trí tuệ nhân tạo | RecursiveChunker (`recursive`) | 58 | 853.2 | Tốt hơn fixed-size vì ưu tiên ranh giới đoạn/câu |

### Strategy Của Tôi

**Loại:** Custom article-level chunking + Hybrid BM25 + Vector retrieval

**Mô tả cách hoạt động:**  
Tôi dùng cấu trúc của văn bản luật để tách tài liệu theo từng `Điều`, thay vì chunk theo độ dài cố định. Mỗi điều trở thành một `Document` riêng, có metadata như `document_type`, `language`, `jurisdiction`, `chapter`, `article`, `article_number`, và `article_title`. Sau đó tôi dùng hybrid retrieval: BM25 bắt các từ khóa pháp lý chính xác, còn vector search bổ sung tín hiệu ngữ nghĩa. Điểm hybrid được tính bằng cách normalize BM25 và vector score về `0..1`, rồi kết hợp mặc định `70% BM25 + 30% vector`.

**Tại sao tôi chọn strategy này cho domain luật/chính sách?**  
Văn bản luật có cấu trúc rất rõ theo chương, điều, khoản, điểm; vì vậy article-level chunking giữ được đơn vị pháp lý tự nhiên. BM25 phù hợp vì câu hỏi pháp lý thường chứa các cụm từ chính xác như "hiệu lực thi hành", "hành vi bị nghiêm cấm", "điều khoản chuyển tiếp". Vector retrieval vẫn hữu ích khi cách hỏi của người dùng không trùng hoàn toàn với từ ngữ trong văn bản.

**Code snippet (nếu custom):**

```python
store.search_hybrid_with_filter(
    query,
    top_k=3,
    metadata_filter={"document_type": "law", "language": "vi"},
    bm25_weight=0.7,
)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Luật Trí tuệ nhân tạo | RecursiveChunker baseline | 58 | 853.2 | Khá tốt, nhưng không biết metadata điều luật |
| Luật Trí tuệ nhân tạo | Article-level + Hybrid BM25/Vector | 35 | Theo từng điều luật | Tốt nhất cho benchmark: top-1 đúng 5/5 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Quân) | Article-level + Hybrid BM25/Vector | 10/10 | Bám cấu trúc Điều luật, BM25 bắt đúng cụm từ pháp lý và vector search bổ sung semantic match. | Phụ thuộc heading/metadata rõ ràng; cần tokenization tiếng Việt tốt hơn nếu corpus lớn. |
| Nam | ArticleChunker (1 Điều/chunk) | 10/10 | Giữ trọn vẹn một Điều luật, đơn giản và rất hợp với dữ liệu pháp lý. | Chunk dài có thể trả về context rộng hơn câu hỏi nhỏ. |
| Đạt | RecursiveChunker (chunk_size=400) | 10/10 (Local top-3) | Chunk nhỏ gọn, linh hoạt, ít phụ thuộc metadata. | Một Điều dài có thể bị chia thành nhiều mảnh nên thiếu ngữ cảnh pháp lý đầy đủ. |
| Khoa | ArticleChunker + 8 file + chapter filter | 8/10 (Local top-1) | Metadata `chapter` giảm nhiễu mạnh, nhất là các câu hỏi có phạm vi chương rõ ràng. | Q1 về định nghĩa vẫn miss Điều 3; cần tách nhỏ Điều có nhiều định nghĩa hoặc dùng embedder tốt hơn. |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
> Nhóm strategy theo Điều luật là phù hợp nhất cho domain văn bản pháp luật vì mỗi Điều là một đơn vị ngữ nghĩa và pháp lý hoàn chỉnh. Trong các biến thể đã thử, hướng **Article-level + metadata/filter + hybrid BM25/vector** mạnh nhất cho các query pháp lý có từ khóa chính xác: BM25 giúp bắt đúng tiêu đề/thuật ngữ, vector search giúp khi cách hỏi khác cách viết trong luật, còn metadata filter giảm nhiễu giữa các chương.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk` — approach:**  
Tôi dùng regex `(?<=[.!?])\s+` để tách văn bản tại ranh giới câu sau dấu `.`, `!`, hoặc `?`, rồi gom tối đa `max_sentences_per_chunk` câu vào một chunk. Hàm xử lý chuỗi rỗng bằng cách trả về list rỗng và strip khoảng trắng để chunk gọn hơn.

**`RecursiveChunker.chunk` / `_split` — approach:**  
Recursive chunking thử tách theo thứ tự separator lớn đến nhỏ: đoạn trống, xuống dòng, câu, khoảng trắng, rồi fallback cắt ký tự. Base case là khi đoạn hiện tại đã nhỏ hơn hoặc bằng `chunk_size`. Cách này giúp giữ ngữ cảnh tự nhiên tốt hơn fixed-size vì ưu tiên ranh giới văn bản có ý nghĩa.

### EmbeddingStore

**`add_documents` + `search` — approach:**  
Mỗi `Document` được lưu thành record gồm `id`, `doc_id`, `content`, `metadata`, `embedding`, token BM25 và token tiêu đề. Vector search embed query, tính dot product giữa query embedding và document embedding, rồi sort giảm dần theo score.

**`search_with_filter` + `delete_document` — approach:**  
Metadata filter được áp dụng trước khi search, đúng với yêu cầu của retrieval có filter. `delete_document` loại bỏ tất cả record có `metadata["doc_id"]` khớp với doc_id cần xóa và trả về `True` nếu có record bị xóa.

**BM25 + Hybrid retrieval — approach:**  
BM25 token hóa văn bản bằng regex lowercase word tokens, tính document frequency và average document length tại thời điểm search. Với hybrid retrieval, tôi normalize BM25 score và vector score về `0..1`, sau đó kết hợp theo trọng số mặc định `bm25_weight=0.7`. Kết quả hybrid trả thêm `bm25_score` và `vector_score` để dễ phân tích.

### KnowledgeBaseAgent

**`answer` — approach:**  
Agent nhận `retrieval_strategy` gồm `vector`, `bm25`, hoặc `hybrid`. Khi trả lời, agent retrieve top-k chunk liên quan, build prompt gồm câu hỏi và retrieved context, rồi gọi `llm_fn`. Prompt yêu cầu chỉ trả lời dựa trên context và nói rõ nếu context không đủ hỗ trợ.

### Test Results

```text
pytest tests/ -q
54 passed, 2 warnings in 0.15s
```

**Số tests pass:** 54 / 54

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is used for machine learning. | Python is commonly used to train AI models. | high | 0.9171 | Có |
| 2 | Vector stores retrieve similar embeddings. | A vector database searches documents by semantic similarity. | high | 0.9454 | Có |
| 3 | AI systems must protect personal data. | Artificial intelligence applications should safeguard private information. | high | 0.8585 | Có |
| 4 | The policy sets compliance deadlines. | The law defines transition periods for compliance. | high | 0.8547 | Có |
| 5 | Bananas are yellow fruit. | A firewall blocks unauthorized network traffic. | low | 0.8265 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Pair 5 bất ngờ nhất vì hai câu không liên quan về mặt nghĩa nhưng điểm vẫn khá cao trong phép thử đơn giản này. Điều này nhắc rằng kết quả phụ thuộc mạnh vào cách tạo vector; mock/simple vectors không thể hiện ngữ nghĩa tốt như embedding model thật, nên khi đánh giá retrieval cần xem trực tiếp chunk trả về chứ không chỉ nhìn score.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries trên implementation cá nhân với strategy: article-level chunks + hybrid BM25/vector + metadata filter `{"document_type": "law", "language": "vi"}`.

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Luật Trí tuệ nhân tạo có hiệu lực từ ngày nào? | Luật có hiệu lực thi hành từ ngày 01 tháng 3 năm 2026, trừ các nội dung liên quan quy định tại Điều 35. |
| 2 | Những hành vi nào bị nghiêm cấm khi sử dụng hệ thống trí tuệ nhân tạo? | Các hành vi bị nghiêm cấm gồm lợi dụng/chiếm đoạt hệ thống AI để vi phạm pháp luật; lừa dối, thao túng, lợi dụng nhóm dễ tổn thương; tạo/phổ biến nội dung giả mạo nguy hại; xử lý dữ liệu trái luật; cản trở giám sát; che giấu thông tin bắt buộc; và lợi dụng nghiên cứu/thử nghiệm/kiểm định để làm trái luật. |
| 3 | Khung đạo đức trí tuệ nhân tạo quốc gia dựa trên những nguyên tắc nào? | Dựa trên an toàn, tin cậy, không gây hại; tôn trọng quyền con người/quyền công dân; công bằng, minh bạch, không phân biệt đối xử; thúc đẩy hạnh phúc, thịnh vượng, phát triển bền vững; khuyến khích đổi mới sáng tạo và trách nhiệm xã hội. |
| 4 | Cơ quan nào là đầu mối quản lý nhà nước về trí tuệ nhân tạo? | Bộ Khoa học và Công nghệ là cơ quan đầu mối, chịu trách nhiệm trước Chính phủ thực hiện quản lý nhà nước về trí tuệ nhân tạo trên phạm vi cả nước. |
| 5 | Điều khoản chuyển tiếp quy định thời hạn tuân thủ nào cho hệ thống trí tuệ nhân tạo đã hoạt động trước ngày luật có hiệu lực? | 18 tháng từ ngày luật có hiệu lực đối với hệ thống AI trong y tế, giáo dục và tài chính; 12 tháng đối với các hệ thống không thuộc nhóm đó. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Hiệu lực của luật | Điều 34 - Hiệu lực thi hành | 1.000 | Có | Dựa vào Điều 34: hiệu lực từ 01/03/2026, trừ nội dung liên quan Điều 35 |
| 2 | Hành vi bị nghiêm cấm | Điều 7 - Các hành vi bị nghiêm cấm | 0.831 | Có | Nêu các nhóm hành vi bị cấm trong sử dụng/phát triển/triển khai AI |
| 3 | Khung đạo đức AI quốc gia | Điều 26 - Khung đạo đức trí tuệ nhân tạo quốc gia | 0.893 | Có | Tóm tắt các nguyên tắc an toàn, quyền con người, công bằng, minh bạch, phát triển bền vững |
| 4 | Cơ quan đầu mối quản lý nhà nước | Điều 30 - Nội dung và trách nhiệm quản lý nhà nước về trí tuệ nhân tạo | 0.987 | Có | Xác định Bộ Khoa học và Công nghệ là cơ quan đầu mối |
| 5 | Điều khoản chuyển tiếp/thời hạn tuân thủ | Điều 35 - Điều khoản chuyển tiếp | 0.871 | Có | Nêu thời hạn 18 tháng cho y tế/giáo dục/tài chính và 12 tháng cho hệ thống khác |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
> Tôi học được từ Khoa cách tách dữ liệu thành 8 file theo chương và dùng metadata `chapter` để thu hẹp retrieval space. Từ Nam, tôi thấy ArticleChunker thuần rất hiệu quả với văn bản luật vì giữ nguyên một Điều. Từ Đạt, tôi học được RecursiveChunker vẫn là baseline linh hoạt khi metadata chưa đủ tốt.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
> Nhóm khác dùng cách thiết kế benchmark queries có gold answer rõ ràng và kiểm tra top-3 relevant thay vì chỉ nhìn similarity score. Cách này giúp đánh giá RAG thực tế hơn, vì score cao chưa chắc câu trả lời đã đúng và có grounding tốt.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ chuẩn hóa encoding và làm sạch Markdown/HTML kỹ hơn ngay từ đầu, vì file luật có nhiều ký tự và HTML table gây nhiễu khi đọc bằng terminal. Tôi cũng sẽ thiết kế metadata theo cấu trúc pháp lý rõ hơn, ví dụ thêm `chapter_number`, `clause_numbers`, và có thể tách sâu hơn theo khoản nếu câu hỏi cần trả lời chi tiết.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **99 / 100** |
