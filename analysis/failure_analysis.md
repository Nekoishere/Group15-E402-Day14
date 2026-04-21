# Failure Analysis Report

## Executive Summary
Hệ thống VinLex đã trải qua quá trình đo lường tự động nhằm so sánh Agent Version 1 (`VinLex-Day14`) và Agent Version 2 (`VinLex-Day14-V2`). Kết quả cho thấy Agent V2 cải thiện rõ rệt ở pha Generation: `Avg Judge Score` tăng `+0.8409`, `Pass Rate` tăng `+15.15%`. Tuy nhiên, pha Retrieval hiện là nút thắt cổ chai lớn nhất khi `Hit Rate` chỉ đạt `25.0%`.

Vì vậy, quyết định Release cho V2 hiện tại là **ROLLBACK**. Lý do chính gồm:
- `Hit Rate` thấp hơn ngưỡng yêu cầu `50%`
- `Cost` tăng mạnh, vượt xa budget cho phép (`+~70%` so với V1)

Nói ngắn gọn: **V2 trả lời tốt hơn khi có ngữ cảnh, nhưng hệ thống vẫn chưa tìm được đúng ngữ cảnh đủ ổn định để release an toàn.**

---

## Benchmark Comparison
- **Agent V1 (`VinLex-Day14`)**: Phiên bản cơ sở chuẩn ban đầu.
- **Agent V2 (`VinLex-Day14-V2`)**: Phiên bản tối ưu prompt và logic sinh câu trả lời ở pha Generation.

Phiên bản V2 cải thiện mạnh khả năng sử dụng ngữ cảnh để trả lời, nâng điểm trung bình từ `2.8636` lên `3.7045`. Tuy nhiên, về mặt Retrieval, cả hai agent đang dùng chung một cấu hình chunking chưa hợp lý, nên cùng chịu một mức `Hit Rate` thấp là `25.0%`.

---

## Metric Table

| Metric | V1 | V2 | Delta | Phân Tích Hiện Trạng |
| :--- | :--- | :--- | :--- | :--- |
| **Avg Judge Score** | 2.8636 | 3.7045 | +0.8409 | V2 sinh câu trả lời tốt hơn rõ rệt nhờ prompt optimization |
| **Pass Rate** | 66.67% | 81.82% | +15.15% | Tỷ lệ pass tăng mạnh nhờ V2 xử lý generation tốt hơn |
| **Agreement Rate** | 42.42% | 26.52% | -15.90% | Độ đồng thuận giữa các judge giảm, cho thấy mâu thuẫn đánh giá tăng lên |
| **Hit Rate** | 25.00% | 25.00% | 0.00% | Retrieval đang rất yếu, 75% số case rơi vào zero-hit |
| **Cost (USD)** | $0.0070 | $0.0119 | +$0.0049 | Chi phí của V2 tăng gần 70%, vượt xa budget 10% |

---

## Trust Analysis

### Mối tương quan giữa Retrieval và Answer Quality
- Khi `Hit = 1`, tỷ lệ `Pass` là **100.0%**.
- Khi `Hit = 0`, hệ thống vẫn pass `34` case nhưng fail `11` case còn lại.
- `Zero-Hit` xuất hiện ở `45/60` case, tương đương `75%`.

Điều này cho thấy:
- Nếu Retrieval hoạt động đúng, hệ thống đáng tin cậy.
- Nếu Retrieval thất bại, model vẫn có thể trả lời “trông có vẻ đúng” nhờ pretraining knowledge, nhưng đó là **độ đúng may mắn**, không phải độ đúng có căn cứ tài liệu.

### Kết luận về độ tin cậy
Hệ thống hiện **chưa đủ chuẩn để tin cậy ở tầng Retrieval**. Khi không truy xuất được đúng nguồn, LLM buộc phải “mò mẫm” theo kiến thức nền, làm rủi ro hallucination tăng cao, đặc biệt nguy hiểm với bài toán quy chế học vụ.

---

## Risk Analysis

### 1. Hallucination Risk
`Zero-Hit` đạt `75.0%`, nghĩa là ở phần lớn case, hệ thống không lấy được đúng chunk nguồn. Với bài toán liên quan tới quy định học vụ, điểm số, thủ tục và mốc thời gian, đây là rủi ro rất nghiêm trọng vì câu trả lời sai có thể gây hậu quả trực tiếp cho người dùng.

### 2. Cost Risk
V2 tiêu thụ nhiều token hơn, kéo chi phí từ `$0.0070` lên `$0.0119`. Mức tăng này gần `70%`, vượt xa ngưỡng `10%` mà Release Gate cho phép. Đây là rào cản lớn nếu muốn scale hệ thống ra production với lượng user cao hơn.

### 3. Agreement Drop
`Agreement Rate` giảm từ `42.42%` xuống `26.52%`. Điều này cảnh báo rằng hai judge model đang đánh giá V2 thiếu đồng nhất hơn trước. Đây chưa chắc là lỗi của V2, nhưng là tín hiệu cho thấy prompt đánh giá hoặc tiêu chí consensus còn cần hiệu chuẩn lại.

---

## Root Cause Analysis - 5 Whys

### Vấn đề
`Hit Rate` chỉ đạt `25.0%`, khiến Retrieval trở thành bottleneck nghiêm trọng nhất của toàn hệ thống.

1. **Tại sao Hit Rate chỉ còn 25.0%?**  
   Vì trong phần lớn trường hợp, ChromaDB trả về chunk không chứa đúng thông tin cốt lõi cần để trả lời câu hỏi.

2. **Tại sao ChromaDB truy xuất sai chunk?**  
   Vì khoảng cách cosine giữa vector câu hỏi và vector chunk đúng quá xa, nên chunk cần thiết không được xếp hạng cao.

3. **Tại sao vector bị lệch xa như vậy?**  
   Vì cấu hình chunking trong `config.py` đang để `CHUNK_SIZE_CHARS` quá lớn (`1800` ký tự), làm một chunk chứa quá nhiều ý khác nhau.

4. **Tại sao chunk quá dài lại làm Retrieval yếu đi?**  
   Vì xuất hiện hiện tượng **vector dilution**: embedding của chunk bị pha loãng bởi quá nhiều nội dung, khiến các fact nhỏ, sắc nét trong câu hỏi không còn match tốt với chunk chứa đáp án thật.

5. **Root cause là gì?**  
   → **Chunking Strategy đang sai kích thước và sai mức độ focus.**  
   Chunk quá lớn làm hỏng khả năng truy xuất fact-level question answering, dẫn tới retrieval sai dù dữ liệu gốc vẫn tồn tại trong kho tài liệu.

---

## Recommendation

### Với Retrieval
- Giảm `CHUNK_SIZE_CHARS` xuống khoảng `500 - 800`
- Giảm `CHUNK_OVERLAP_CHARS` về mức vừa phải để vẫn giữ được continuity nhưng không làm chunk trùng lặp quá mức
- Re-ingest toàn bộ tài liệu sau khi đổi cấu hình

### Với Cost
- Rút gọn prompt V2, loại bỏ các chỉ dẫn dư thừa
- Cân nhắc dùng model rẻ hơn cho một số bước reasoning phụ hoặc fallback path
- Giảm số lần gọi tie-breaker nếu có thể chuẩn hóa prompt judge tốt hơn

### Với Multi-Judge Consensus
- Chuẩn hóa lại evaluation criteria
- Làm rõ rubric thế nào là “partial pass”, “mostly correct”, “complete answer”
- Rà soát prompt để giảm conflict cực đoan giữa các judge

---

## Next Action
1. Chấp hành quyết định **ROLLBACK**. Không release V2 ở trạng thái hiện tại.
2. Sửa `CHUNK_SIZE_CHARS` trong [config.py](/abs/path/c:/Users/Administrator/Desktop/code/Group15-E402-Day14/config.py).
3. Xóa index vector cũ trong thư mục ChromaDB.
4. Chạy lại ingestion pipeline bằng `python ingest_pdfs.py`.
5. Chạy lại benchmark bằng `python main.py` cho cả V1 và V2.
6. Kỳ vọng sau khi sửa chunking:
   - `Hit Rate > 50%`
   - giảm zero-hit rõ rệt
   - Release Gate có cơ hội chuyển từ `ROLLBACK` sang `APPROVE`
