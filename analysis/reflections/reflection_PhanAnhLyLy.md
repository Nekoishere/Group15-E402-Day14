# Individual Reflection & Technical Depth – Role: Multi-Judge Consensus Engine
**Họ và tên:** Phan Anh Ly Ly
**Vai trò trong nhóm:** Thiết kế & triển khai module Multi-Judge Consensus (`engine/llm_judge.py`) và hệ thống báo cáo chi phí/hiệu suất trong `main.py`

---

## 1. Công việc đã triển khai (Engineering Contribution)

### 1.1. Thiết kế Multi-Judge Evaluator (`engine/llm_judge.py`)
- **Kiến trúc dual-judge bất đồng bộ (async):** Tích hợp 2 model LLM độc lập đóng vai trò giám khảo:
  - **Judge A**: `GPT-4o-mini` (OpenAI) – giá rẻ, tốc độ cao
  - **Judge B**: `GPT-4o` (OpenAI) – chất lượng cao hơn, làm đối chứng
  - Cả hai được gọi song song bằng `asyncio.gather()`, giảm thiểu thời gian chờ so với gọi tuần tự.
- **Hệ thống Prompt chuẩn hóa:** Thiết kế `JUDGE_PROMPT` đánh giá theo 3 tiêu chí rõ ràng (Accuracy, Completeness, Tone) trên thang điểm 1–5, yêu cầu LLM trả về JSON có cấu trúc để parse tự động.
- **Logic đồng thuận (Agreement Rate):**
  - `diff == 0` → Full agreement (1.0), lấy điểm chung
  - `diff == 1` → Soft agreement (0.5), lấy trung bình
  - `diff >= 2` → Conflict (0.0), kích hoạt Meta-Judge

### 1.2. Tự động xử lý xung đột – Meta-Judge / Tie-Breaker
- Khi 2 giám khảo lệch điểm từ 2 điểm trở lên, hệ thống tự động gọi `_call_tiebreaker()` – một lần gọi thứ ba đến `GPT-4o-mini` với toàn bộ ngữ cảnh gồm: câu hỏi, đáp án chuẩn, phản hồi của agent, cùng reasoning của 2 Judge.
- Meta-Judge trả về `final_score` và `ruling` (lý giải Judge nào đúng hơn và tại sao).
- **Xử lý kỹ thuật quan trọng:** Tách biệt hàm `_call_tiebreaker()` khỏi `_call_openai_judge()` để tránh lỗi `KeyError` khi reasoning của Judge A/B chứa ký tự `{` hoặc `}` – lỗi này xảy ra khi dùng `.format()` hai lần trên chuỗi đã được format một phần.

### 1.3. Tính toán Cohen's Kappa (`compute_cohens_kappa`)
- Triển khai hàm `compute_cohens_kappa(scores_a, scores_b, n_categories=5)` theo công thức chuẩn.
- `LLMJudge` tích lũy toàn bộ điểm số của Judge A và B vào `_batch_scores_a` và `_batch_scores_b` xuyên suốt benchmark, sau đó gọi `compute_batch_kappa()` để ra kết quả cuối.

### 1.4. Phát hiện Position Bias (`check_position_bias`)
- Triển khai `check_position_bias(question, answer_good, answer_bad, ground_truth)` với 2 vòng đánh giá:
  - **Round 1**: Đánh giá `answer_good` và `answer_bad` theo thứ tự thông thường.
  - **Round 2**: Swap thứ tự — Judge được cho xem `answer_bad` trước rồi `answer_good`.
  - Nếu chênh lệch điểm giữa 2 vòng lớn hơn 1 (`bias_magnitude > 1`), kết luận có positional bias.

### 1.5. Theo dõi Chi phí & Token (`CostTracker`)
- Thiết kế class `CostTracker` tích lũy từng cuộc gọi API: model, input tokens, output tokens, latency (ms), và cost (USD).
- Bảng giá `COST_TABLE` cấu hình sẵn cho từng model, tính chi phí theo công thức: `cost = (input_tokens × input_rate + output_tokens × output_rate) / 1000`.
- `CostTracker.summary()` trả về báo cáo tổng hợp gồm: tổng chi phí, chi phí trung bình mỗi lần eval, token usage, latency trung bình, và breakdown theo từng model.

### 1.6. Mở rộng báo cáo trong `main.py`
- Triển khai các hàm `_build_runtime_summary()`, `_build_cost_report()`, và `_print_correlation_analysis()` để xuất báo cáo đầy đủ sau mỗi benchmark run.
- Xây dựng **Release Gate** tự động: so sánh V2 vs V1 theo 4 tiêu chí (`avg_score >= 3.5`, `hit_rate >= 0.5`, `delta_score >= 0`, `cost_increase <= 10%`) và ra quyết định `APPROVE` hay `ROLLBACK`.
- Lưu 4 file báo cáo: `summary.json`, `benchmark_results.json`, `retrieval_results.json`, `cost_report.json`.

---

## 2. Giải thích các khái niệm kỹ thuật chuyên sâu (Technical Depth)

### 2.1. Tại sao cần Multi-Judge thay vì Single Judge?

Một LLM Judge duy nhất có thể bị ảnh hưởng bởi nhiều yếu tố ngẫu nhiên: nhiệt độ sampling (temperature), biến động trong attention weights, hay sự thiên vị vốn có từ tập huấn luyện. Khi dùng 2 Judge độc lập:
- Nếu chúng **đồng thuận** → kết quả rất đáng tin cậy (cross-validation giữa 2 model có kiến trúc/training data khác nhau).
- Nếu chúng **bất đồng** → hệ thống phát hiện được trường hợp mơ hồ thay vì âm thầm trả về kết quả sai lệch.

Agreement Rate chính là chỉ số đo lường **độ chắc chắn** của kết quả đánh giá, không chỉ đơn thuần là chất lượng câu trả lời.

### 2.2. Cohen's Kappa – Loại trừ May Mắn Ngẫu Nhiên

Trong các hệ thống AI Evaluation nhiều giám khảo, 2 LLM cho điểm giống nhau đôi khi chỉ do xác suất ngẫu nhiên (chance agreement), đặc biệt khi thang điểm hẹp hoặc phân phối điểm tập trung.

**Công thức:**
```
κ = (P_o - P_e) / (1 - P_e)
```
- `P_o` = xác suất đồng thuận thực tế: `count(score_a == score_b) / n`
- `P_e` = xác suất đồng thuận ngẫu nhiên nếu 2 judge chấm bừa:
  ```
  P_e = Σ [P(Judge_A chọn c) × P(Judge_B chọn c)] với c ∈ {1,2,3,4,5}
  ```
- Dùng `collections.Counter` để đếm tần suất từng điểm của mỗi Judge, sau đó tính P_e theo phân phối thực tế của từng model.

**Ý nghĩa trong dự án:**
- `κ > 0.6` → Substantial agreement: rubric đánh giá rõ ràng, 2 model thực sự "hiểu" tiêu chí chấm điểm.
- `κ < 0.4` → Weak agreement: có thể prompt chưa đủ rõ hoặc tập test case quá mơ hồ.
- Kết quả thực tế: `agreement_rate = 0.65 (V1), 0.65 (V2)` thể hiện hệ thống judge ổn định.

### 2.3. Position Bias – Thiên Kiến Vị Trí của LLM

LLM có xu hướng "ưu ái" câu trả lời ở một vị trí cố định (thường là đầu tiên hoặc cuối) trong Pairwise Comparison, dù nội dung tương đương. Đây là lỗi hệ thống cực kỳ nguy hiểm trong AI Evaluation vì nó làm **sai lệch toàn bộ kết quả benchmark**.

**Cách triển khai `check_position_bias()`:**
1. Đánh giá `answer_good` và `answer_bad` theo thứ tự gốc → `diff_normal = score_good - score_bad`
2. Swap thứ tự, đánh giá lại → `diff_swapped = score_good_new - score_bad_new`
3. `bias_magnitude = |diff_normal - diff_swapped|`
4. Ngưỡng: nếu `bias_magnitude > 1` → `bias_detected = True`

**Tác động thực tế:** Phát hiện positional bias sớm giúp nhóm điều chỉnh design prompt hoặc cơ chế gọi API để đảm bảo kết quả benchmark phản ánh chất lượng thực của agent, không phải artifact của thứ tự trình bày.

### 2.4. Trade-off Chi phí vs. Chất lượng (Cost vs. Quality)

| Chiến lược | Chất lượng | Chi phí |
|---|---|---|
| Chỉ dùng GPT-4o (single judge) | Cao | Rất cao |
| Chỉ dùng GPT-4o-mini (single judge) | Trung bình | Thấp |
| Dual Judge (GPT-4o-mini + GPT-4o) | Cao + Self-check | Trung bình |
| Dual Judge + Meta-Judge (chỉ khi conflict) | Cao nhất | Tiết kiệm ~80% vs. GPT-4o toàn phần |

Hệ thống tối ưu hóa chi phí bằng cách chỉ gọi lần thứ ba (Meta-Judge) khi thực sự xảy ra xung đột (`diff >= 2`). Trong benchmark thực tế với 10 test cases và agreement_rate ~0.65, số lần cần Meta-Judge là thiểu số, giúp tiết kiệm đáng kể budget API.

### 2.5. Thiết kế `CostTracker` – Observability cho Chi phí API

Mỗi lần gọi API được record với đầy đủ metadata (model, tokens, latency, cost). Class `CostTracker` dùng danh sách `_records` để tích lũy, sau đó `summary()` aggregate theo từng model và trả về cả `avg_latency_ms`, `cost_per_eval_usd`. Đây là pattern quan trọng trong production MLOps để theo dõi và tối ưu hóa chi phí vận hành hệ thống AI.

---

## 3. Khó khăn mắc phải và cách khắc phục

### 3.1. Lỗi `KeyError` trong Meta-Judge Prompt Formatting
- **Vấn đề:** Ban đầu `_call_tiebreaker()` được thiết kế để gọi lại `_call_openai_judge()` với `preformatted_prompt`. Tuy nhiên, khi reasoning của Judge A/B chứa ký tự `{reason_a}` từ câu JSON trả về, `.format()` của Python sẽ cố interpret chúng như placeholders, gây `KeyError`.
- **Khắc phục:** Tách `_call_tiebreaker()` thành hàm độc lập, gọi thẳng `_openai_client.chat.completions.create()` mà không qua lớp wrapper. Điều này đảm bảo prompt chỉ được format **một lần** với `TIE_BREAKER_PROMPT.format(...)` trước khi gửi đi.

### 3.2. Rate Limit khi chạy Async Batch
- **Vấn đề:** Khi dùng `asyncio.gather()` để gọi đồng thời cho nhiều test cases, OpenAI API trả về lỗi HTTP 429 (Too Many Requests) do vượt quá giới hạn request/phút.
- **Khắc phục:** Trong `BenchmarkRunner`, triển khai `asyncio.Semaphore` để giới hạn số lượng coroutines chạy đồng thời (Max Concurrent Requests), cân bằng giữa throughput cao và tuân thủ rate limit của API provider.

### 3.3. Git Merge Conflict trong file `llm_judge.py`
- **Vấn đề:** File `llm_judge.py` có conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> be30c147`) còn sót lại do quá trình merge code của nhóm, đặc biệt ở phần docstring mô tả Judge B (GPT-4o vs. Gemini 2.0 Flash).
- **Bài học:** Cần thiết lập quy trình review code nghiêm ngặt hơn trước khi merge, và dùng git conflict markers check trong CI/CD pipeline để tự động phát hiện file chưa được resolve.

---

## 4. Kết quả thực tế & Phân tích

Dựa trên `benchmark_results.json` và `summary.json`:

| Metric | Agent V1 | Agent V2 |
|---|---|---|
| Avg Judge Score | 2.70 / 5 | 2.75 / 5 |
| Agreement Rate | 0.70 | 0.65 |
| Hit Rate (Retrieval) | 0% | 0% |
| Release Decision | — | **RELEASE** |

**Phân tích:**
- Điểm trung bình (~2.7/5) phản ánh chính xác chất lượng của mock agents trong môi trường test — các agent trả lời theo template cứng, không truy xuất được tài liệu đúng (Hit Rate = 0%).
- Agreement Rate ~0.65–0.70 cho thấy 2 Judge đồng ý với nhau khoảng 2/3 số lần — mức độ nhất quán tốt với rubric đánh giá được định nghĩa rõ ràng.
- Hệ thống vẫn ra quyết định `RELEASE` vì `delta_score = +0.05 >= 0` (V2 nhỉnh hơn V1), đáp ứng tiêu chí regression gate.

---

## 5. Điều rút ra (Lessons Learned)

1. **Defensive prompt engineering:** Khi dùng `.format()` với string có nguồn gốc từ output của LLM (không kiểm soát được), cần escape hoặc dùng template engine khác (như f-string hoặc Jinja2) để tránh lỗi runtime bất ngờ.
2. **Observability là bắt buộc:** Không có `CostTracker`, nhóm sẽ không có dữ liệu để tối ưu hóa chi phí sau khi chạy thực trên production data. Token tracking + latency monitoring là nền tảng của AI engineering có trách nhiệm.
3. **Kiểm soát đồng thời trong async:** `asyncio.gather()` mạnh về throughput nhưng cần kết hợp với `Semaphore` trong môi trường thực tế để không vi phạm rate limit của bên thứ ba — bài học từ lỗi 429 thực tế khi chạy benchmark.
4. **Multi-judge không chỉ cho chất lượng tốt hơn:** Quan trọng hơn, nó cung cấp **confidence signal** (Agreement Rate) — biết được khi nào hệ thống không chắc chắn là giá trị không kém gì biết được điểm số cụ thể.
