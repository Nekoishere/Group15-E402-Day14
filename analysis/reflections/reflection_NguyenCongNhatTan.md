# Individual Reflection - Lab 14

**Họ và tên:** Nguyễn Công Nhật Tân  
**MSSV:** 2A202600141  
**Vai trò trong Lab:** Data / AI Engineer  

---

## 1. Engineering Contribution (15 điểm)
**Những đóng góp cụ thể vào các module của hệ thống:**
- **Golden Dataset Generation (`data/synthetic_gen.py`):** Viết script kết nối trực tiếp với ChromaDB bằng thư viện `chromadb` để lôi xuất toàn bộ chunk IDs thật. Từ đó sử dụng `AsyncOpenAI` (`gpt-4o-mini`) để sinh song song hơn 60 QA pairs. Điều này giải quyết bài toán mock data ban đầu, giúp hệ thống có ground truth map đúng 1-1 với Vector database để chấm điểm chân thực. Mình cũng tự thêm các ca "red-teaming" thủ công như prompt injection hay out-of-context.
- **Retrieval Pipeline (`engine/retrieval_eval.py`):** Nâng cấp logic `evaluate_batch()` từ chạy tuần tự sang chạy đồng thời (concurrent) bằng `asyncio.gather` cùng với `Semaphore(10)` để bóp luồng chống Rate Limit của OpenAI. Nhờ thay đổi này, việc test 66 cases giảm từ >7 phút xuống chỉ còn dưới nửa phút.
- Xử lý các hard-cases thủ công, bỏ qua (skip) logic tính Hit Rate đối với câu hỏi mang tính Adversarial do không có ground truth.

*(Note: Commit có thể được kiểm chứng trên nhánh main, các commit về "Refactor synthetic_gen to use ChromaDB chunks" và "Implement concurrent evaluate_batch").*

---

## 2. Technical Depth (15 điểm)
Trong quá trình làm, mình phải áp dụng kiến thức chuyên sâu để giải quyết các quyết định khó thuật toán:

* **MRR (Mean Reciprocal Rank) vs Hit Rate:** Hit Rate chỉ là hệ số nhị phân (có/không trong top K). Nó không nói lên trải nghiệm của Agent. Nếu tài liệu đúng nằm ở top 5, Hit Rate vẫn 100%, nhưng LLM có thể bị nhiễu do đọc phải 4 tài liệu kia trước. MRR giải quyết bài toán này bằng cách phạt vị trí (1/rank). Thấy rõ trong log: Hit Rate là 5% nhưng MRR chỉ 0.026, chứng tỏ các chunk đúng nếu có xuất hiện cũng thường nằm ở cuối danh sách.
* **Cohen's Kappa & Position Bias:** Khi build Judge LLM, dùng 2 model (Ví dụ GPT-4o và Claude) dễ bị ảo tưởng vì đôi khi chúng "ngẫu nhiên" chấm giống nhau. Cohen's Kappa loại trừ tỷ lệ đồng thuận ngẫu nhiên, cho thấy mức độ đáng tin cậy *thực sự*. Position Bias thì xuất hiện khi đưa Input A, B cho model so sánh — model luôn thích A hơn. Cách mình làm là bắt buộc chạy thêm test case đảo B lên trước để chéo kiểm.
* **Trade-off Chi phí và Chất lượng (Cost/Quality):** Mọi người thường dùng GPT-4o cho mọi thứ. Ở hệ thống của nhóm, quy trình được tối ưu bằng cách: Dùng `text-embedding-3-small` (cực rẻ, $0.02/1M tokens) để nhúng vector + `gpt-4o-mini` để rà soát sơ lược. Chỉ gọi Judge bằng `gpt-4o` ở bước cuối. Việc này giúp giảm hơn 80% cost cho khâu eval để bù đắp cho việc mua chất lượng ở khâu đánh giá chéo.

---

## 3. Problem Solving (10 điểm)
**Vấn đề lớn nhất nhóm gặp phải:** *Hit Rate thực tế đo được chỉ đạt 5.0%, quá 95% câu hỏi bị Zero-Hit.*
* **Quá trình điều tra:** Ban đầu mình tưởng do lỗi code tính toán hoặc so khớp string ID sai. Tuy nhiên, sau khi viết script cô lập lệnh `agent._rag.retrieve()` và in trực tiếp biến `distance`, mình phát hiện agent trả về id khác hoàn toàn, với khoảng cách Cosine trên ngưỡng.
* **Nguyên nhân cốt lõi (Root Cause):** Khảo sát cấu trúc hệ thống, mình phát hiện `config.py` thiết lập `CHUNK_SIZE_CHARS = 1800`. Việc nén nửa trang giấy vào 1 vector gây ra hiện tượng *Vector Dilution* — ý nghĩa bị loãng. Khi query một câu ngắn (10 từ), Vector Query gọn gàng bị "lạc" khi đo Cosine Similarity với Vector "thùng rác" 1800 ký tự.
* **Cách giải quyết:**
  1. Chứng minh được nguyên nhân lỗi, báo cáo vào `failure_analysis.md`.
  2. Mình đã trực tiếp thay cấu hình `CHUNK_SIZE_CHARS` xuống 800, `CHUNK_OVERLAP` xuống 80, và chuẩn bị chạy lại file `ingest_pdfs.py` để clear database, chứng minh giải pháp tối ưu hệ thống triệt để.
