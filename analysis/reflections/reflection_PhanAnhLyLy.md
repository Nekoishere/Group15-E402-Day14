# Individual Reflection & Technical Depth - Role: Multi-Judge Consensus
**Họ và tên:** Phan Anh Ly Ly
**Vai trò trong nhóm:** Xây dựng module Multi-Judge Consensus & Evaluator

## 1. Công việc đã triển khai (Engineering Contribution)
- **Thiết kế Multi-Judge Evaluator (`engine/llm_judge.py`)**: 
  - Tích hợp 2 model độc lập cùng đóng vai trò giám khảo là `GPT-4o-mini` (Judge A) và `Gemini-2.5-Flash` (Judge B) chạy bất đồng bộ (async) để tăng tốc độ xử lý.
  - Xử lý logic đồng thuận (Agreement Rate): Trả về 1.0 nếu 2 giám khảo cho điểm bằng nhau, 0.5 nếu chênh lệch 1 điểm.
- **Tự động xử lý xung đột (Conflict Resolution)**:
  - Khi 2 giám khảo cho điểm lệch nhau từ 2 điểm trở lên (ví dụ: GPT cho 2 điểm, Gemini cho 5 điểm), hệ thống sẽ bật cơ chế **Meta-Judge (Tie-breaker)**. 
  - Meta-Judge sẽ phân tích cả 2 feedback của Judge A và B cùng câu trả lời gốc để đưa ra phán quyết (score cuối cùng và ruling).
- **Mở rộng Metrics báo cáo (`main.py`)**: 
  - Tính toán Cost API, báo cáo số lượng Token sử dụng và độ trễ (Latency).
  - Kết hợp logic tính Cohen's Kappa trong suốt toàn bộ quá trình Benchmark.

---

## 2. Giải thích các khái niệm kỹ thuật chuyên sâu (Technical Depth)

### 2.1. Độ đồng thuận qua Cohen's Kappa
Trong các hệ thống AI Evaluation nhiều giám khảo, nếu 2 LLM cho điểm giống nhau, điều đó cũng có thể là do xác suất ngẫu nhiên (chance agreement). 
- **Cohen's Kappa (κ)** giải quyết vấn đề này bằng cách loại trừ yếu tố "ngẫu nhiên" ra khỏi độ đồng thuận. 
- Công thức: `κ = (P_o - P_e) / (1 - P_e)`, trong đó:
  - `P_o` là xác suất đồng thuận thực tế quan sát được của GPT và Gemini.
  - `P_e` là xác suất đồng thuận do ngẫu nhiên (nếu 2 model tự ý cho điểm bừa).
- **Ý nghĩa trong dự án**: Score của Cohen's Kappa > 0.6 chứng tỏ bộ Prompt đánh giá của chúng ta rất rõ ràng và 2 model thực sự "hiểu" rubrics chứ không phải đồng thuận ngẫu nhiên.

### 2.2. Position Bias (Thiên kiến vị trí của LLM Judge)
Khi dùng LLM làm Judge (đặc biệt trong phương pháp Pairwise Comparison - so sánh 2 câu trả lời A và B), LLM thường có xu hướng "ưu ái" câu trả lời xuất hiện ở vị trí đầu tiên (A) hoặc đôi khi là vị trí cuối (B) dù chất lượng như nhau.
- **Cách theo dõi & xử lý**: Tôi đã thiết kế hàm `check_position_bias()` chạy 2 round. Round 1: So sánh [Answer 1, Answer 2]. Round 2: Cố tình swap thứ tự thành [Answer 2, Answer 1]. Nếu LLM thay đổi điểm/lựa chọn đáng kể sau khi đảo thứ tự, chứng tỏ có Position Bias. 
- **Tác động**: Yếu tố này giúp hệ thống Benchmark của nhóm khách quan hơn, hạn chế sai số cực lớn từ chính framework giám khảo.

### 2.3. Trade-off giữa Chi phí và Chất lượng (Cost vs Quality)
- **Vấn đề**: Sử dụng GPT-4o cho 100% test cases sẽ đem lại độ chính xác rất cao nhưng đẩy chi phí `Cost/eval` cao gấp 10-20 lần so với model nhỏ.
- **Giải pháp áp dụng**: Tôi kết hợp `GPT-4o-mini` (rẻ) và `Gemini Flash` (thường được free tier/giá cực thấp). Chỉ khi nào xảy ra tranh cãi (conflict > 1), hệ thống mới gọi lên `Meta-Judge` (có thể config dùng GPT-4o lớn) để tiết kiệm tokens.
- **Kết quả**: Vẫn giữ được độ tin cậy tương đương 1 giám khảo xịn (single strict judge) nhưng tiết kiệm hơn ~80% quỹ tiền API.

---

## 3. Khó khăn mắc phải và cách khắc phục
- **Vấn đề Async/ Rate Limit**: Khi dùng asyncio để gọi 50 test cases, Open AI API và Gemini đều bị giới hạn Rate Limit gây ra lỗi HTTP 429.
- **Khắc phục**: Tôi đã triển khai sử dụng `asyncio.Semaphore` (giới hạn Max Concurrent) nhằm khống chế số lượng request đồng thời tới API của LLM, giúp toàn trình chạy trơn tru mà không bị lỗi nghẽn cổ chai mạng.
