# Individual Reflection & Technical Depth - Role: Regression Release Gate & DevOps
**Họ và tên:** Phan Nguyễn Việt Nhân
**Vai trò trong nhóm:** Xây dựng module Regression Testing, Release Gate tự động & Debug Pipeline

---

## 1. Công việc đã triển khai (Engineering Contribution)

### 1.1. Regression Testing Framework (`main.py`)
- **Thiết kế V1 vs V2 Comparison**: Tách biệt hai phiên bản Agent thực sự khác nhau thay vì dùng chung một instance. `Agent_V1_Base` sử dụng cấu hình gốc (temperature=0.2, không có instruction bổ sung), còn `Agent_V2_Optimized` dùng temperature=0 và thêm một *structured-response prompt addon* yêu cầu format câu trả lời rõ ràng hơn, trích dẫn nguồn cụ thể hơn.
- **Delta Analysis đa chiều**: So sánh V1 và V2 trên 4 chiều đồng thời — `Avg Judge Score`, `Pass Rate`, `Agreement Rate`, và `Cost (USD)` — thay vì chỉ so sánh điểm trung bình như trước.
- **Truyền cấu hình qua chuỗi**: Triển khai cơ chế override config qua constructor (`generation_temperature`, `prompt_addon`) xuyên suốt `MainAgent` → `VinLexChatbot` → `RAGPipeline`, đảm bảo V2 thực sự sử dụng cấu hình khác biệt mà không cần sửa global config.

### 1.2. Automatic Release Gate (`main.py`)
- Viết logic tự động ra quyết định "APPROVE" hoặc "ROLLBACK" dựa trên **4 ngưỡng cứng (hard thresholds)**:
  1. `avg_score >= 3.5` — Chất lượng câu trả lời tối thiểu
  2. `hit_rate >= 50%` — Retrieval phải đủ tin cậy trước khi release
  3. `delta_score >= 0.0` — V2 không được tệ hơn V1 (chống regression)
  4. `cost_increase <= 10%` — Chi phí không được vượt ngân sách
- Khi Gate bị block, hệ thống tự động liệt kê **từng lý do cụ thể** (blocking reasons) để dev team biết chính xác cần cải thiện điều gì.

### 1.3. Cost Tracking & Bug Fixes
- **Sửa bug Cost Tracking**: `reports/summary.json` luôn báo `total_cost_usd: 0` vì `runner.py` không truyền cost vào kết quả. Sửa bằng cách gọi `LLMJudge.get_cost_summary()` trực tiếp sau mỗi benchmark run, đồng thời thêm `LLMJudge.reset_cost_tracker()` để đo chi phí V1 và V2 riêng biệt.
- **Sửa bug `TypeError` trong `LLMJudge`**: `main.py` gọi `LLMJudge(model="gpt-4o-mini")` nhưng constructor không nhận tham số → crash ngay khi khởi động. Sửa thành `LLMJudge()`.
- **Sửa bug `KeyError` trong `_call_tiebreaker`**: Hàm này format `TIE_BREAKER_PROMPT` rồi truyền kết quả vào `_call_openai_judge`, hàm này lại gọi `.format()` lần nữa trên string đã format → Python tìm thấy các `{...}` từ nội dung judge reasons (ví dụ `{\n  description}`) và throw `KeyError`. Sửa bằng cách cho `_call_tiebreaker` gọi trực tiếp OpenAI API thay vì đi qua `_call_openai_judge`.

---

## 2. Giải thích các khái niệm kỹ thuật chuyên sâu (Technical Depth)

### 2.1. Regression Testing trong AI System là gì?

Trong phần mềm truyền thống, Regression Testing kiểm tra xem tính năng cũ có bị vỡ sau khi thêm code mới hay không. Trong AI System, khái niệm này phức tạp hơn vì output không deterministic — cùng một input có thể cho kết quả khác nhau giữa các lần chạy.

**Vì vậy, Regression Testing trong AI cần:**
- **Baseline (V1)**: Một phiên bản Agent đã được đánh giá và có metrics làm tham chiếu.
- **Challenger (V2)**: Phiên bản mới với thay đổi cụ thể (prompt, model, threshold).
- **Statistical comparison**: So sánh phân phối điểm, không chỉ so giá trị điểm trung bình đơn lẻ.

**Điều quan trọng**: V1 và V2 *phải* thực sự khác nhau về cấu hình — dùng cùng một agent cho cả hai (như phiên bản lỗi trước đây) chỉ đo *noise* từ LLM temperature, không phải sự khác biệt thực sự của model.

### 2.2. Release Gate và ý nghĩa của từng ngưỡng

Release Gate là cơ chế tự động ngăn một phiên bản AI tệ hơn được deploy lên production. Mỗi ngưỡng có lý do riêng:

| Ngưỡng | Giá trị | Lý do |
|--------|---------|-------|
| `min_score = 3.5` | Tuyệt đối | Dưới 3.5/5 = câu trả lời thường thiếu sót hoặc sai — không đủ tiêu chuẩn user-facing |
| `min_hit_rate = 0.50` | Tuyệt đối | Nếu Retrieval chỉ tìm được đúng chunk < 50% cases, mọi Generation tốt chỉ là hallucination may mắn |
| `min_delta_score = 0` | Tương đối | Đây là ngưỡng anti-regression: V2 không được kém hơn V1, kể cả chỉ -0.001 |
| `max_cost_increase = 10%` | Tương đối | Kiểm soát "quality creep" — dùng model đắt hơn chỉ để tăng 0.1 điểm là không hợp lý |

**Insight quan trọng**: Khi cả 4 ngưỡng đều fail (như kết quả hiện tại: score 3.1 < 3.5, hit_rate 31.67% < 50%), Root Cause thực sự nằm ở tầng *Chunking Strategy* chứ không phải Prompt hay Model — vì dù Generation có tốt đến đâu cũng không cứu được Retrieval tệ.

### 2.3. Trade-off giữa Version Isolation và Chi phí Benchmark

Một thách thức trong Regression Testing là: nếu chạy đầy đủ cả V1 lẫn V2 benchmark (mỗi version 66 test cases × 2 LLM judges + retrieval eval), chi phí API tăng gấp 3 lần so với chỉ benchmark một version.

**Quyết định thiết kế của tôi**:
- **Retrieval Evaluation**: Chạy **một lần duy nhất** dùng V1 agent. Vì V1 và V2 có cùng retrieval config (chỉ khác generation), kết quả retrieval được dùng chung cho cả hai → tiết kiệm ~33% API call.
- **Judge Cost Reset**: Dùng `LLMJudge.reset_cost_tracker()` trước mỗi benchmark run để đo chi phí V1 và V2 *riêng biệt*, cho phép Release Gate so sánh `cost_v1` vs `cost_v2` một cách chính xác.
- **Cost per eval**: Metric này quan trọng hơn tổng chi phí — nếu V2 cải thiện score nhưng tốn gấp đôi chi phí mỗi eval, đó là trade-off cần cân nhắc kỹ trước khi deploy.

---

## 3. Khó khăn mắc phải và cách khắc phục

### Bug 1: Double `.format()` trong tiebreaker gây `KeyError`
**Mô tả**: `_call_tiebreaker` format `TIE_BREAKER_PROMPT` (chứa `{{final_score}}`), sau đó pass chuỗi đã format vào `_call_openai_judge` — hàm này lại gọi `.format()` một lần nữa. Lần format thứ hai tìm thấy `{...}` từ nội dung của judge reasons (ví dụ model trả về reason có dạng `{\n  description: ...}`) và throw `KeyError: '\n  description'`.

**Khắc phục**: Cho `_call_tiebreaker` gọi trực tiếp `openai_client.chat.completions.create()` thay vì routing qua `_call_openai_judge`. Chỉ có **một lần** `.format()` duy nhất, không có double-processing. Thêm comment giải thích lý do để tránh tái phát trong tương lai.

### Bug 2: V1 và V2 dùng cùng agent instance
**Mô tả**: Code gốc dùng `agent = MainAgent()` và pass cùng object này cho cả V1 lẫn V2. Kết quả: `delta_score` chỉ là noise từ LLM temperature, không phản ánh bất kỳ cải tiến thực sự nào. Regression Test trở nên vô nghĩa.

**Khắc phục**: Thiết kế lại `MainAgent` để nhận `generation_temperature` và `prompt_addon` qua constructor, sau đó tạo `MainAgentV2` subclass với cấu hình cụ thể (temperature=0, structured-response instructions). V2 giờ có hành vi thực sự khác V1, tạo ra delta score có ý nghĩa thống kê.

### Bug 3: Gemini model 404
**Mô tả**: Model `gemini-2.5-flash-preview-04-17` là bản preview từ 2025, đã bị deprecate → Judge B luôn trả về fallback score = 3, làm lệch toàn bộ agreement rate và consensus metrics.

**Khắc phục**: Cập nhật sang `gemini-2.0-flash` (GA model, stable) và đồng bộ tên model trong toàn bộ `COST_TABLE`, return dicts, docstrings để tránh mismatch giữa model name và cost lookup.
