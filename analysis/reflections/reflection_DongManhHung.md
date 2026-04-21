# Individual Reflection & Technical Depth - Role: Performance (Async) & Observability
**Họ và tên:** Đồng Mạnh Hùng
**Vai trò trong nhóm:** Tối ưu hiệu năng pipeline benchmark, thiết kế concurrency control và báo cáo Cost/Token usage

---

## 1. Công việc đã triển khai (Engineering Contribution)

### 1.1. Tối ưu Async cho toàn pipeline benchmark
- **Refactor `BenchmarkRunner` (`engine/runner.py`)**: Chuyển từ cách chạy theo từng `batch_size=5` tuần tự sang mô hình `asyncio.Semaphore + asyncio.gather` phủ toàn dataset. Điểm khác biệt quan trọng là pipeline luôn giữ đủ số request đang chạy, không phải chờ xong batch cũ mới mở batch mới.
- **Tách cấu hình concurrency ra `config.py`**: Thêm `BENCHMARK_MAX_CONCURRENCY` và `RETRIEVAL_EVAL_CONCURRENCY` để nhóm có thể tune tốc độ/rate limit mà không phải sửa logic code. Đây là bước cần thiết để đạt rubric phần Performance thay vì hardcode.
- **Đồng bộ Retrieval Eval với Benchmark Runner**: `engine/retrieval_eval.py` cũng dùng semaphore cấu hình được, nên cả retrieval stage lẫn multi-judge stage đều chạy song song có kiểm soát. Nhờ vậy pipeline end-to-end không còn điểm nghẽn tuần tự lớn.

### 1.2. Bổ sung observability cho runtime
- **Đo wall-clock runtime theo stage (`main.py`)**: Tôi thêm thống kê runtime riêng cho `retrieval_eval`, `Agent_V1_Base`, `Agent_V2_Optimized` và `total_pipeline_sec`. Đây là chỉ số rubric yêu cầu để chứng minh pipeline đủ nhanh cho 50 cases.
- **Theo dõi throughput thực tế**: Trong phần `performance` của mỗi benchmark, hệ thống ghi `cases_per_minute`, `avg/min/max latency`. Chỉ nhìn latency từng case là chưa đủ, vì mục tiêu async là tối ưu throughput toàn pipeline.
- **Đưa cờ tự kiểm tra rubric vào report**: Trường `meets_async_target_under_2_min_for_50_cases` trong `reports/summary.json`/`reports/cost_report.json` giúp nhóm chứng minh trực tiếp điều kiện `< 2 phút cho 50 cases`.

### 1.3. Báo cáo Cost & Token usage chi tiết
- **Nâng cấp `CostTracker` (`engine/llm_judge.py`)**: Không chỉ giữ `total_cost_usd`, tôi mở rộng sang `total_input_tokens`, `total_output_tokens`, `total_tokens`, `avg_latency_ms_per_call`, và breakdown theo từng model.
- **Thêm report mới `reports/cost_report.json`**: File này gom đầy đủ runtime + token + cost cho từng benchmark version (V1/V2), kèm `llm_call_records` để drill down nếu cần phân tích model nào đang tốn chi phí bất thường.
- **Sửa cách tính `cost_per_eval_usd`**: Trước đây cost đang chia theo số lượng LLM calls, làm metric bị lệch vì 1 test case có thể phát sinh 2 judge calls hoặc thêm tie-breaker. Tôi đổi sang chia theo số lượng eval cases để metric này đúng ý nghĩa vận hành.

---

## 2. Giải thích các khái niệm kỹ thuật chuyên sâu (Technical Depth)

### 2.1. Vì sao batch tuần tự không tối ưu bằng semaphore concurrency?

Khi dùng vòng lặp kiểu:
- lấy 5 cases,
- `await gather(...)`,
- chờ xong hết,
- rồi mới mở 5 cases tiếp theo,

thì toàn pipeline bị rơi vào trạng thái **barrier synchronization**. Chỉ cần 1 request chậm trong batch là 4 request nhanh còn lại cũng phải đứng chờ. Điều này làm giảm throughput rõ rệt.

Semaphore concurrency tốt hơn vì:
- luôn giữ tối đa `N` request active,
- case nào xong sớm sẽ nhường slot ngay cho case tiếp theo,
- tổng thời gian gần với `critical path` thực hơn, thay vì bị khóa bởi batch chậm nhất.

Nói ngắn gọn: **batching giới hạn theo đợt**, còn **semaphore giới hạn theo số lượng request đồng thời thực sự**.

### 2.2. Wall-clock time khác gì latency từng request?

- **Latency**: thời gian xử lý của một case riêng lẻ.
- **Wall-clock runtime**: thời gian người dùng thực sự phải đợi để chạy xong toàn bộ benchmark.

Trong hệ thống async, mục tiêu chính là giảm **wall-clock runtime** chứ không nhất thiết giảm latency của từng request. Một request judge vẫn có thể mất 2-3 giây, nhưng nếu 10 request chạy cùng lúc thì 50 cases vẫn hoàn thành rất nhanh. Vì vậy báo cáo hiệu năng phải có cả:
- latency distribution,
- throughput (`cases_per_minute`),
- tổng runtime từng stage.

### 2.3. Trade-off giữa concurrency và rate limit

Concurrency càng cao không đồng nghĩa càng tốt. Nếu mở quá nhiều request đồng thời:
- OpenAI/Gemini có thể trả `429 Too Many Requests`,
- số lần retry tăng lên,
- wall-clock runtime thực tế có thể còn tệ hơn.

Vì vậy tôi chọn cách:
- để concurrency thành config,
- dùng semaphore để khóa trần số request,
- đo runtime thật trong report để nhóm có cơ sở tune dần.

Đây là tư duy **performance engineering thực tế**: tối ưu dựa trên đo lường, không tối ưu theo cảm giác.

---

## 3. Khó khăn mắc phải và cách khắc phục

### Vấn đề 1: Runner đang "có async nhưng chưa thật sự nhanh"
**Mô tả**: Code cũ có `asyncio.gather`, nhưng vẫn đặt trong vòng lặp batch tuần tự. Về mặt cú pháp là async, nhưng về mặt hệ thống vẫn còn điểm nghẽn lớn.

**Khắc phục**: Tôi refactor lại `run_all()` để spawn task cho toàn bộ dataset và dùng `Semaphore` giới hạn concurrency. Cách này vừa giữ được song song hóa, vừa tránh bắn request vô hạn.

### Vấn đề 2: Metric cost/eval bị hiểu sai
**Mô tả**: Nếu lấy `total_cost / total_calls`, số liệu sẽ bị méo vì 1 eval case thường tương ứng nhiều LLM calls (2 judges, có lúc thêm tiebreaker). Khi đó `cost_per_eval_usd` không phản ánh chi phí thực sự của 1 test case.

**Khắc phục**: Tôi sửa tracker để nhận `eval_count` từ benchmark run, sau đó tính `cost_per_eval_usd` theo số lượng test cases. Đồng thời vẫn giữ `total_calls` để người đọc phân biệt được cost vận hành và cost ở mức API call.

### Vấn đề 3: Thiếu report đủ chi tiết để chấm phần Performance
**Mô tả**: `summary.json` cũ chỉ có tổng cost/tokens cơ bản, chưa đủ để chứng minh pipeline chạy nhanh và chưa đủ sâu để phân tích tối ưu chi phí.

**Khắc phục**: Tôi bổ sung `performance` vào từng benchmark summary và xuất thêm `reports/cost_report.json` chứa:
- runtime theo stage,
- token input/output/total,
- cost per eval,
- latency trung bình mỗi LLM call,
- breakdown theo model,
- call-level records để debug sâu khi cần.

Nhờ đó phần Performance (Async) của nhóm không chỉ "nói là nhanh" mà có số liệu định lượng rõ ràng để bảo vệ trước rubric.

---

## 4. Đánh giá kết quả benchmark thực tế

Từ lần benchmark hiện tại, tôi rút ra 3 kết luận vận hành quan trọng:

- **V2 tốt hơn rõ rệt ở Generation**: `Avg Judge Score` tăng từ `2.8636` lên `3.7045`, còn `Pass Rate` tăng từ `66.67%` lên `81.82%`. Điều này xác nhận giả thuyết ban đầu của tôi là prompt optimization và cấu trúc trả lời có tác dụng thật.
- **Retrieval mới là bottleneck hệ thống**: Dù V2 mạnh hơn ở khâu sinh câu trả lời, `Hit Rate` của cả V1 và V2 đều chỉ đạt `25.0%`. Điều đó cho thấy nút nghẽn không nằm ở async runner hay generation prompt, mà nằm ở chunking/retrieval design.
- **Hiệu năng tốt nhưng chưa đủ để release nếu chất lượng nền chưa đạt**: Một pipeline có thể chạy nhanh, có cost report đẹp, nhưng nếu `Hit Rate` còn dưới ngưỡng và cost tăng gần `70%` thì hệ thống vẫn phải `ROLLBACK`. Đây là điểm rất quan trọng trong AI Engineering: performance optimization không thể thay thế correctness ở tầng dữ liệu.

Vì vậy, nếu nhìn dưới góc độ cá nhân của tôi phụ trách phần Performance, thành công lớn nhất không chỉ là làm benchmark chạy song song hơn, mà là giúp nhóm **đo được rất rõ**: V2 đang tốt lên ở đâu, đang fail ở đâu, và fail vì nguyên nhân nào. Không có lớp observability này thì nhóm sẽ rất dễ tối ưu sai chỗ.
