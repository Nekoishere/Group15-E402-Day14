# Failure Analysis Report (Phân tích lỗi hệ thống)

## 1. Mối liên hệ giữa Retrieval Quality và Answer Quality

Dựa trên dữ liệu chạy thật của tập Golden Dataset (66 cases), hệ thống thu được kết quả Correlation như sau:
- **Hit Rate tổng thể rất thấp (5.0%)**, với 57/60 trường hợp (95%) rơi vào trạng thái "Zero-Hit" (ChromaDB không tìm được chunk đúng chứa nội dung gốc).

**Sự tương quan (Dựa trên mô hình chấm ảo hiện tại):**
- **Hit=1 & Pass (3 cases):** Khi Retrieval hoạt động tốt, hệ thống luôn trả về câu trả lời đúng (Tỷ lệ = 100%).
- **Hit=0 & Pass (57 cases):** Retrieval thất bại (không tìm đúng chunk chính xác) nhưng câu trả lời vẫn vượt qua mức điểm trung bình ảo. Tuy nhiên, trên thực tế, khi Hit Rate = 0, GPT bị thiếu ngữ cảnh (Context) trầm trọng và dễ dẫn đến tình trạng **Hallucination** (AI tự bịa câu trả lời theo kiến thức chung thay vì quy chế của trường).

**Kết luận:** Điểm nghẽn (Bottleneck) chí mạng của hệ thống VinLex hiện tại nằm ở pha **Retrieval**, không phải Generation.

---

## 2. Phân tích "5 Whys" (Root Cause Analysis cho lỗi Hit Rate 5%)

Vấn đề: *Hệ thống tìm kiếm (Retrieval) thất bại trong việc tìm tài liệu đúng cho 95% câu hỏi.*

1. **Tại sao hệ thống không tìm được tài liệu gốc?**
   → Vì khi dùng Cosine Distance để đo độ tương đồng trong ChromaDB, Vector của câu hỏi bị lệch so với Vector của đoạn văn bản (Chunk) chứa câu trả lời.
   
2. **Tại sao Vector lại bị lệch xa (Distant) nhau như vậy?**
   → Vì độ dài tài liệu của một Chunk là cực kỳ lớn. Cụ thể, xem trong `config.py`, hệ thống đang ép cấu hình `CHUNK_SIZE_CHARS = 1800` (dài bằng gần nửa trang giấy A4 chứa hàng chục ý khác nhau).
   
3. **Tại sao Chunk dài 1800 ký tự lại làm hỏng độ đo Cosine?**
   → Vì khi nén một lượng văn bản khổng lồ (hỗn tạp nhiều chủ đề) vào một Embedding duy nhất, ý nghĩa của Vector đó bị "pha loãng" (hiện tượng Vector Dilution). 
   
4. **Tại sao Vector loãng lại gây ra tìm kiếm sai?**
   → Trong khi Vector của tài liệu bị pha loãng, câu hỏi của Dataset lại cực kỳ ngắn và cụ thể (ví dụ: *"SGPA là gì?"*). Thuật toán embedding `text-embedding-3-small` sẽ khó tìm thấy sự đồng điệu (Cosine Similarity cao) giữa một ý nhỏ xíu và một đoạn văn khổng lồ.
   
5. **Root cause (Nguyên nhân gốc rễ) là gì?**
   → **Chiến lược Chunking (Chunking Strategy) bị thiết kế sai kích thước.** 1800 ký tự là quá to cho công tác truy xuất thông tin cụ thể (Fact-QA). ChromaDB đã vô tình bốc nhầm các chunk có chứa một vài "từ khóa" trùng lặp ngẫu nhiên thay vì chunk chứa định nghĩa gốc. 

### 💡 Đề xuất khắc phục (Action Item):
Cần Rollback Agent lại và sửa lại tham số ở `config.py`:
- Giảm `CHUNK_SIZE_CHARS` xuống khoảng `500 - 800` ký tự.
- Xóa Data ChromaDB cũ, chạy lại file `ingest_pdfs.py` để nhúng (embed) lại dữ liệu. Hit Rate dự kiến sẽ tăng vọt lên mức 80%+.
