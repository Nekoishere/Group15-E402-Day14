# Individual Reflection - Lab 14

**Họ và tên:** [Your Name]  
**MSSV:** [Your Student ID]  
**Vai trò trong Lab:** AI / Backend Integration Engineer  

---

## 1. Engineering Contribution (15 điểm)

Trong Lab Day 14, phần việc chính mình phụ trách là **tích hợp agent đã xây ở Day 06 vào pipeline benchmark của Day 14**, thay thế phần `MainAgent` đang là mock bằng runtime RAG thật để hệ thống đánh giá có thể chạy trên output thực tế thay vì dữ liệu giả lập.

- **Di chuyển runtime cần thiết từ Day 06 sang Day 14:** Mình chọn cách chỉ copy các module thực sự phục vụ suy luận RAG, gồm `config.py`, `backend/chatbot.py`, `backend/rag.py`, `backend/vector_store.py` và dữ liệu `data/chroma_db`. Mình không copy phần Flask app, template hay session management vì các thành phần đó không phục vụ benchmark mà chỉ làm Day 14 phình to và khó debug hơn. Cách làm này giúp repo Day 14 giữ được tính self-contained nhưng vẫn tái sử dụng đúng lõi kỹ thuật đã làm ở Day 06.

- **Bổ sung dependency để code có thể chạy trong Day 14:** Sau khi chuyển code, mình cập nhật `requirements.txt` để thêm `chromadb`. Đây là bước nhỏ nhưng quan trọng, vì nếu chỉ copy source mà không đồng bộ dependency thì pipeline sẽ fail ngay ở bước import. Ở góc nhìn engineering, integration không chỉ là “copy file”, mà còn phải bảo đảm runtime environment phù hợp với code vừa tích hợp.

- **Sửa retrieval output để phục vụ evaluation:** Trong Day 06, `VectorStore.query()` trả về `text`, `metadata` và `distance`, nhưng chưa trả về `chunk id`. Đối với Day 14, đây là thiếu sót quan trọng vì retrieval evaluation cần `retrieved_ids` để tính **Hit Rate** và **MRR**. Mình đã chỉnh lại `backend/vector_store.py` để kết quả query trả về thêm trường `id`. Việc này giúp liên kết retrieval runtime với benchmark metric một cách trực tiếp, thay vì chỉ đánh giá chất lượng answer ở mức bề mặt.

- **Wire lại `agent/main_agent.py` theo interface của Day 14:** Day 14 yêu cầu một class `MainAgent` có hàm `async def query(question: str) -> dict`. Trong khi đó, agent Day 06 được thiết kế cho Flask route và chạy đồng bộ. Mình đã viết lại `agent/main_agent.py` thành một lớp wrapper bất đồng bộ, dùng `asyncio.to_thread(...)` để chạy chatbot đồng bộ của Day 06 trong worker thread. Hàm này chuẩn hóa output về đúng schema benchmark cần dùng:
  - `answer`
  - `contexts`
  - `retrieved_ids`
  - `metadata`

- **Thêm cơ chế degrade an toàn khi lỗi:** Trong quá trình wire agent, mình chủ động bao `try/except` trong `_query_sync()`. Mục tiêu là nếu API key sai, import lỗi, hoặc vector DB có vấn đề, benchmark không chết cả pipeline mà vẫn nhận được một response có `query_type = "error"` để nhóm tiếp tục điều tra. Đây là tư duy quan trọng khi làm evaluation system: pipeline cần quan sát được lỗi, không chỉ “crash im lặng”.

- **Kiểm tra tính hợp lệ ở mức syntax trước khi chạy benchmark:** Sau khi tích hợp, mình dùng `python -m py_compile` để xác minh các file mới không lỗi cú pháp. Bước xác minh sớm này giúp giảm thời gian debug vòng sau, đặc biệt khi đang làm integration giữa hai codebase khác nhau.

---

## 2. Technical Depth (15 điểm)

### 2.1. Vì sao không import trực tiếp Day 06 từ đường dẫn cũ?

Một cách nhanh là để Day 14 import thẳng code ở thư mục Day 06 bằng path tuyệt đối. Tuy nhiên, cách này tạo ra dependency ngầm giữa hai repo bài tập:

- Day 14 không còn độc lập khi nộp bài
- Máy khác hoặc CI khác rất dễ fail vì path không tồn tại
- Việc tái hiện kết quả benchmark trở nên khó kiểm soát

Vì vậy mình chọn cách **copy runtime cần thiết vào chính repo Day 14**, giữ submission self-contained. Đây là trade-off tốt hơn giữa tốc độ tích hợp và tính ổn định khi chấm bài.

### 2.2. Vì sao cần `asyncio.to_thread(...)`?

`BenchmarkRunner` của Day 14 gọi `await self.agent.query(...)`, nghĩa là interface agent bắt buộc phải là async. Trong khi đó, VinLex runtime của Day 06 là code đồng bộ, bên trong có các lời gọi OpenAI client đồng bộ và logic xử lý tuần tự.

Nếu mình giữ nguyên code sync và gọi trực tiếp trong `query()`, event loop của benchmark sẽ bị block. Khi đó, dù phần runner có dùng `asyncio.gather`, throughput toàn pipeline vẫn kém vì mỗi agent call chặn event loop.

`asyncio.to_thread(...)` giải quyết đúng điểm này:

- Giữ nguyên code Day 06, không cần refactor toàn bộ sang async
- Cho phép benchmark runner vẫn gọi agent bằng giao diện async
- Tách blocking workload ra thread worker, không chặn event loop chính

Đây là một kỹ thuật integration khá thực tế: thay vì rewrite hệ thống cũ quá sâu, mình dùng một lớp adapter để tương thích với runtime mới.

### 2.3. Vì sao `retrieved_ids` quan trọng hơn chỉ trả về `contexts`?

Nếu agent chỉ trả về context text, ta có thể đọc thủ công để cảm nhận retrieval “có vẻ đúng”. Nhưng benchmark tự động thì không thể dựa vào cảm giác. Nó cần một định danh rõ ràng để so sánh với ground truth.

- **Hit Rate** kiểm tra xem ít nhất một ground-truth chunk có xuất hiện trong top-k retrieval hay không
- **MRR** đo vị trí xuất hiện đầu tiên của chunk đúng, tức là chunk đúng nằm càng cao thì điểm càng tốt

Cả hai metric này đều cần danh sách `retrieved_ids`. Vì vậy, bước sửa `VectorStore.query()` để trả về `id` không phải là chi tiết phụ, mà là điều kiện bắt buộc để Day 14 đánh giá retrieval đúng theo rubric.

### 2.4. Chuẩn hóa output agent có ý nghĩa gì?

Ở Day 06, output được thiết kế cho web app:
- `answer`
- `query_type`
- `sources`
- các cờ redirect/contact

Ở Day 14, output phải phục vụ benchmark engine:
- answer để judge generation
- contexts để trace nguồn
- retrieved_ids để tính retrieval metrics
- metadata để ghi model, sources, trạng thái đặc biệt

Việc mình viết lớp wrapper để chuẩn hóa output giúp tách **business logic của agent** ra khỏi **evaluation contract**. Đây là một pattern rất quan trọng trong AI Engineering: cùng một model runtime có thể được dùng cho UI, API hoặc benchmark, miễn là có adapter phù hợp ở mỗi lớp.

---

## 3. Problem Solving (10 điểm)

### Vấn đề 1: Agent Day 06 và benchmark Day 14 không cùng interface

**Mô tả:** Day 06 được xây như một chatbot web đồng bộ, còn Day 14 cần một agent async để runner gọi hàng loạt test case. Nếu copy nguyên file vào thì code không chạy đúng vai trò benchmark.

**Cách giải quyết:** Mình không sửa sâu vào logic VinLex, mà tạo `MainAgent` như một adapter layer. Lớp này giữ API của Day 14 nhưng gọi runtime Day 06 ở bên dưới. Cách này giảm rủi ro phá vỡ agent gốc và giúp nhóm tích hợp nhanh hơn.

### Vấn đề 2: Thiếu retrieval IDs nên không thể tính metric retrieval thật

**Mô tả:** Day 06 ưu tiên trải nghiệm chat nên không expose `chunk id` ở đầu ra retrieval. Điều này đủ cho UI hiển thị, nhưng không đủ cho benchmark.

**Cách giải quyết:** Mình truy vết từ `pdf_manager` và xác nhận mỗi chunk đã có ID ngay từ lúc index, sau đó chỉnh `VectorStore.query()` trả thêm `id`. Nhờ đó, agent Day 14 có thể cung cấp `retrieved_ids` cho downstream evaluation.

### Vấn đề 3: Nguy cơ copy quá nhiều code thừa từ Day 06

**Mô tả:** Nếu copy toàn bộ prototype, Day 14 sẽ kéo theo Flask app, static assets, login flow và các thành phần không liên quan. Điều này làm tăng độ phức tạp khi debug và tăng khả năng lỗi môi trường.

**Cách giải quyết:** Mình chỉ chọn lõi runtime tối thiểu cần cho benchmark. Đây là quyết định thiên về maintainability: ít file hơn, ít dependency ngầm hơn, dễ giải thích phần đóng góp cá nhân hơn.

### Vấn đề 4: Integration xong nhưng pipeline vẫn có thể fail vì môi trường

**Mô tả:** Dù syntax đúng, benchmark vẫn có thể hỏng nếu thiếu `OPENAI_API_KEY` hoặc dependency chưa được cài.

**Cách giải quyết:** Mình thêm dependency cần thiết vào `requirements.txt`, dùng `load_dotenv()` ở agent wrapper, và thêm nhánh trả về lỗi có cấu trúc trong `metadata` để nhóm dễ quan sát nguyên nhân thay vì phải debug từ traceback mơ hồ.

---

## 4. Bài học rút ra

Qua phần việc này, mình học được rằng **integration giữa hai giai đoạn phát triển khác nhau khó hơn viết mới từ đầu ở một số điểm**. Vấn đề chính không nằm ở việc “code có chạy không”, mà ở chỗ:

- output có đúng contract downstream hay không
- dependency có đồng bộ hay không
- repo nộp bài có self-contained hay không
- hệ thống có quan sát được lỗi thay vì fail cứng hay không

Nếu tiếp tục cải tiến, mình muốn làm thêm ba hướng:

1. Bổ sung token usage thật vào `metadata` thay vì để `None`
2. Tránh gọi retrieval hai lần trong agent wrapper bằng cách expose chunk retrieval trực tiếp từ chatbot
3. Tiếp tục nối agent thật này vào evaluator và multi-judge thật, thay cho phần placeholder hiện còn trong `main.py`

Nhìn tổng thể, phần đóng góp của mình tập trung vào **đưa agent thật vào đúng pipeline benchmark**, biến Day 14 từ một scaffold có mock response thành một repo có thể đánh giá trên runtime RAG thật của nhóm.
