# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
>Bốn phản hồi cho thấy temperature không tác động đồng đều lên mọi thành phần của văn bản. Ba mức 0.0, 0.5 và 1.0 đều giữ nguyên chủ đề hang Sơn Đoòng, chỉ đến mức 1.5 mô hình mới chuyển hẳn sang một chủ đề khác — cho thấy nội dung có xác suất áp đảo vẫn ổn định qua một dải temperature khá rộng. Tuy nhiên, ngay trong ba phản hồi cùng chủ đề, các chi tiết thứ cấp đã phân kỳ rõ rệt: chiều dài hang thay đổi giữa "9 km", "hơn 5 km" và "ít nhất 9 km", còn vật so sánh chuyển từ máy bay Boeing 747 sang tòa nhà chọc trời 40 tầng. Điều này phù hợp với cơ chế của tham số: temperature điều chỉnh độ phẳng của phân phối xác suất tại từng bước chọn token, nên những vị trí có một phương án chiếm ưu thế rõ rệt vẫn được giữ lại, trong khi các vị trí có nhiều phương án xấp xỉ nhau về xác suất sẽ phân kỳ sớm hơn nhiều. Đáng lưu ý là các con số mâu thuẫn nhau xuất hiện ngay từ những mức temperature thấp, cho thấy việc hạ tham số này giúp tăng tính ổn định của cấu trúc phản hồi nhưng không bảo đảm tính chính xác của dữ kiện.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
>Giá trị temperature được đề xuất là 0.0–0.3, xuất phát từ yêu cầu về tính nhất quán: cùng một truy vấn phải trả về cùng một nội dung. Thực nghiệm ở Câu 1.1 cho thấy ngay tại các mức temperature thấp, mô hình đã đưa ra những con số mâu thuẫn nhau về cùng một sự việc vốn có đáp án xác định. Trong bối cảnh tư vấn chính sách — thời hạn bảo hành, phí đổi trả, điều kiện hoàn tiền — sai lệch dạng này tạo rủi ro trực tiếp về vận hành và pháp lý, vì hai khách hàng đặt cùng một câu hỏi có thể nhận hai câu trả lời khác nhau. Điều này cũng cho thấy hạ temperature là điều kiện cần nhưng chưa đủ: tham số này cải thiện tính ổn định của cấu trúc phản hồi nhưng không bảo đảm tính chính xác của dữ kiện. Do đó, tính nhất quán cần được bảo đảm ở tầng kiến trúc, chẳng hạn ràng buộc mô hình sinh nội dung dựa trên tài liệu chính sách thay vì tri thức nội tại của nó.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
>Với khối lượng 10.000 người dùng, mỗi người gọi API 3 lần và mỗi lần sinh trung bình 350 token, hệ thống tiêu thụ 10,5 triệu token output mỗi ngày, tương đương 10.500 nghìn token. Áp đơn giá output $0,010 trên 1K token của GPT-4o, chi phí là $105 mỗi ngày, tức khoảng $38.325 mỗi năm; trong khi GPT-4o-mini với đơn giá $0,0006 chỉ tốn $6,30 mỗi ngày, tương đương $2.299 mỗi năm. Như vậy GPT-4o đắt hơn 16,67 lần, chênh lệch khoảng $36.026 mỗi năm. Đáng chú ý là tỷ lệ này giữ nguyên ở cả chiều input ($0,0025 so với $0,00015), cho thấy khoảng cách giá giữa hai model là nhất quán chứ không chỉ tập trung ở một chiều. Trường hợp nên dùng model nhỏ là các truy vấn tra cứu đơn giản. Thực nghiệm với câu hỏi về số tỉnh thành của Việt Nam cho thấy hai model trả về cùng một đáp án, nhưng model nhỏ có độ trễ 1,18 giây so với 3,09 giây, tức nhanh hơn 2,6 lần. Ở loại tác vụ này, chi phí tăng thêm không đem lại giá trị tương ứng về chất lượng, đồng thời còn đánh đổi ngược về độ trễ — người dùng vừa phải trả nhiều hơn vừa phải chờ lâu hơn. Ngược lại, GPT-4o xứng đáng với chi phí khi tổn thất kỳ vọng của một lỗi sai vượt quá khoản chênh lệch $36.000 mỗi năm, chẳng hạn trong sinh mã nguồn triển khai trực tiếp lên production hoặc phân tích văn bản pháp lý, nơi một sai sót có thể gây thiệt hại lớn hơn nhiều lần. Ngoài ra, cần tính đến trường hợp tỷ lệ lỗi của model nhỏ cao đến mức buộc phải gọi lại nhiều lần hoặc bổ sung khâu kiểm tra thủ công; khi đó tổng chi phí thực tế có thể vượt phương án dùng model lớn ngay từ đầu, khiến khoản tiết kiệm trên giấy tờ trở nên vô nghĩa.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
>Hai phản hồi phân hóa trên cả ba phương diện: độ dài, từ vựng và cấu trúc trình bày. Phản hồi theo persona sư phạm sử dụng văn xuôi liên tục cùng các ẩn dụ đời thường như "cuốn sổ cái khổng lồ" và "chuỗi các hộp", gần như không chứa thuật ngữ chuyên ngành. Phản hồi theo persona chuyên gia chuyển sang định dạng liệt kê đánh số với tiêu đề in đậm, sử dụng nguyên các thuật ngữ kỹ thuật: phi tập trung, nút mạng, bất biến, hợp đồng thông minh. Điểm cần nhấn mạnh là system prompt không bổ sung tri thức mới cho mô hình — hiểu biết nền về chủ đề được giữ nguyên trong cả hai trường hợp, yếu tố thay đổi là cơ chế chọn lọc và đóng gói tri thức sẵn có. Đây là cơ sở giải thích hiệu quả chi phí của kỹ thuật persona: chỉ tiêu tốn một lượng token input không đáng kể nhưng điều chỉnh được hành vi mô hình, không đòi hỏi tinh chỉnh tham số hay chuẩn bị tập dữ liệu huấn luyện.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.
**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn thực nghiệm gồm 123 từ và khoảng 556 ký tự. Đếm bằng tiktoken với bộ mã hóa của gpt-4o cho kết quả 155 token, trong khi công thức ước lượng số từ / 0,75 mà Part 1 sử dụng cho 164 token, tức cao hơn thực tế 5,8%; công thức dự phòng len(text) // 4 lại cho 139 token, tức thấp hơn thực tế 10,3%. Từ đó suy ra tỷ lệ thực tế của đoạn văn là 1,26 token trên mỗi từ và khoảng 3,59 ký tự trên mỗi token. Nguyên nhân của sai lệch nằm ở chỗ cả hai công thức ước lượng đều được hiệu chỉnh theo mức nén của tiếng Anh, vốn đạt xấp xỉ 4 ký tự trên một token, trong khi tokenizer nén tiếng Việt kém hơn đáng kể. Điều này bắt nguồn từ việc tập ngữ liệu huấn luyện tokenizer nghiêng hẳn về tiếng Anh: các từ tiếng Anh phổ biến thường được gộp trọn vẹn thành một token, còn tiếng Việt có hệ thống dấu thanh và dấu phụ khiến mỗi ký tự chiếm nhiều byte UTF-8 hơn và thường bị tách rời khỏi nguyên âm gốc thay vì hợp nhất. Hệ quả thực tế là với cùng một lượng nội dung, người dùng tiếng Việt phải trả nhiều chi phí hơn và chạm giới hạn cửa sổ ngữ cảnh sớm hơn so với tiếng Anh.Điểm đáng lưu ý nhất về mặt kỹ thuật là hướng của sai số. Công thức dự phòng ước tính thiếu chứ không thừa, nghĩa là hệ thống sẽ báo cáo chi phí thấp hơn thực tế. Sai số theo hướng này nguy hiểm hơn hướng ngược lại: ước tính dư chỉ gây lo lắng không cần thiết, còn ước tính thiếu tạo cảm giác an toàn sai lệch trong khi chi phí thực đã vượt dự toán. Ở quy mô hàng chục nghìn request mỗi ngày như kịch bản ở Câu 1.3, chênh lệch 10% là một khoản đáng kể.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming phát huy giá trị cao nhất trong các ứng dụng tương tác có người dùng theo dõi trực tiếp và độ dài phản hồi lớn. Cần phân biệt rõ: streaming không rút ngắn tổng thời gian sinh nội dung mà chỉ giảm thời gian tới token đầu tiên, qua đó chuyển khoảng chờ thụ động thành thời gian người dùng đã bắt đầu tiếp nhận thông tin — một cải thiện về trải nghiệm cảm nhận chứ không phải về hiệu năng thực. Khi chạy chatbot thực nghiệm, các phản hồi dài vài trăm từ được hiển thị dần theo từng chunk thay vì xuất hiện thành một khối sau vài giây chờ. Ngược lại, chế độ non-streaming phù hợp hơn trong các kịch bản không có người đọc trực tiếp như xử lý theo lô hoặc gọi từ tầng backend, và đặc biệt trong các trường hợp cần toàn bộ phản hồi trước khi hiển thị nhằm thực hiện kiểm duyệt nội dung hoặc kiểm tra tính hợp lệ của định dạng, bởi nội dung đã được stream ra giao diện thì không thể thu hồi.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Với cơ chế delay cố định, toàn bộ client gặp lỗi tại cùng một thời điểm sẽ thực hiện retry tại cùng một thời điểm, khiến máy chủ tiếp nhận một đợt yêu cầu đồng loạt đúng vào giai đoạn năng lực xử lý suy giảm nhất — hiện tượng được gọi là thundering herd. Chu trình lỗi–chờ–retry đồng bộ này tự duy trì, làm máy chủ không có khoảng thời gian nào để phục hồi. Exponential backoff khắc phục bằng cách phân tán tải theo trục thời gian với độ giãn tăng dần (0,1s → 0,2s → 0,4s), đồng thời hoạt động như một cơ chế phân loại ngầm: lỗi càng kéo dài thì khoảng lùi của client càng lớn, qua đó tự động giảm áp lực với các sự cố nghiêm trọng. Trong triển khai thực tế cần bổ sung jitter — thành phần ngẫu nhiên cộng vào thời gian chờ — bởi nếu mọi client áp dụng cùng một công thức tất định thì tính đồng bộ vẫn được duy trì, chỉ dịch chuyển sang các mốc thời gian xa hơn. Một hạn chế của cách triển khai hiện tại là hàm bắt mọi Exception, bao gồm cả các lỗi xác thực hoặc hết hạn mức vốn không thể tự phục hồi; retry đúng cách cần phân loại mã lỗi và chỉ thử lại với nhóm 429 và 5xx.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Ràng buộc về độ dài là lựa chọn có tác động lớn nhất. Thực nghiệm cho thấy khi không ràng buộc, phản hồi vượt trần max_tokens và bị cắt giữa câu thay vì kết thúc hoàn chỉnh, làm giảm chất lượng đầu ra ở cả phương diện nội dung lẫn hình thức. Về mặt chi phí, đơn giá output cao gấp bốn lần đơn giá input ($0,010 so với $0,0025 trên 1K token với GPT-4o), do đó kiểm soát độ dài phản hồi có hiệu quả tối ưu chi phí cao hơn đáng kể so với việc rút gọn prompt. Việc chỉ định tường minh ngôn ngữ đầu ra cũng cần thiết, bởi khi truy vấn chứa thuật ngữ tiếng Anh, mô hình có xu hướng chuyển ngôn ngữ phản hồi theo ngữ cảnh từ vựng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của hệ thống hiện tại là độ chính xác của thống kê chi phí. Do model sử dụng không thuộc tập mã hóa của tiktoken, hàm count_tokens luôn rơi vào nhánh dự phòng; thực nghiệm tại Câu 2.2 cho thấy nhánh này ước tính thiếu 10,3% so với số token thực, đồng nghĩa giá trị total_cost mà hệ thống báo cáo luôn thấp hơn chi phí thực tế. Sai số theo hướng này nguy hiểm hơn sai số ngược lại, vì nó tạo cảm giác chi phí đang trong tầm kiểm soát. Giải pháp đề xuất: thay thế cơ chế ước lượng bằng việc đọc trực tiếp trường usage trong response của API, cụ thể là prompt_tokens và completion_tokens — đây chính là các giá trị nhà cung cấp sử dụng để tính cước, do đó loại bỏ hoàn toàn sai số ước lượng. Điểm cần lưu ý khi triển khai là ở chế độ stream=True, response mặc định không bao gồm trường usage; cần bổ sung tham số stream_options={"include_usage": True} và trích xuất giá trị này từ chunk cuối cùng của stream.
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
