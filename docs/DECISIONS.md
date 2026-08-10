## 2026-08-07 — Scratch Transformer decoding

Người phụ trách: Thành viên 2

Quyết định:
Giữ nguyên checkpoint, tokenizer và kiến trúc được cung cấp. Sử dụng cấu hình
decoding cố định beam4_lp1.1_nr3.

Cấu hình:

- strategy: beam_search
- num_beams: 4
- length_penalty: 1.1
- no_repeat_ngram_size: 3
- max_new_tokens: 96
- min_new_tokens: 10
- seed: 2026

Căn cứ:
Cấu hình dự kiến trong tài liệu dự án được xác nhận trên validation trước khi
chạy test core. Không sử dụng kết quả test để điều chỉnh cấu hình.

Ảnh hưởng:
Không thay đổi manifest, schema prediction, checkpoint, tokenizer hoặc kiến trúc.

## 2026-08-09 — Bổ sung đánh giá decoding Scratch Transformer

Người phụ trách: Thành viên 2

Vấn đề:
Phần đánh giá decoding kế thừa từ kỳ trước được trưởng nhóm xác nhận là chưa
chính xác, nên cần chạy lại một tập cấu hình cố định để có bằng chứng so sánh.

Quyết định:
Chạy bổ sung ba cấu hình trên `test_smoke_10` rồi `validation_select`:

1. `greedy`: greedy tiêu chuẩn, `num_beams=1`, không chặn lặp n-gram.
2. `beam4_lp0.8_nr3`: beam 4, length penalty 0.8, no-repeat n-gram 3.
3. `beam4_lp1.0_nr3`: beam 4, length penalty 1.0, no-repeat n-gram 3.

Cấu hình chung:
`max_new_tokens=96`, `min_new_tokens=10`, seed 2026, FP32 mặc định.

Nguyên tắc:
- Giữ nguyên checkpoint, tokenizer, kiến trúc và manifest.
- Không ghi đè prediction `test_core_2000` đã bàn giao.
- Kết quả bổ sung được so sánh trên validation; không dùng test để lựa chọn.
- Cả ba cấu hình được chốt trước khi xem kết quả bổ sung.

## 2026-08-09 — Thiết kế đánh giá sâu của Thành viên 4

Người phụ trách: Thành viên 4

Quyết định:

- Lấy 200 mẫu test bằng bốn tầng theo hạng độ dài source, 50 mẫu mỗi tầng,
  seed 2026; tập dự phòng 100 là subset cố định của tập 200.
- Human evaluation dùng 50 mẫu và một người chấm; thêm 10 sample block lặp ẩn
  để đo intra-rater consistency, không báo inter-rater agreement.
- Dùng API tương thích OpenAI Chat Completions. Model, API key và base URL được
  cung cấp qua biến môi trường; không khóa provider hoặc model trong code.
- Candidate judge chấm một candidate mỗi request, không nhận reference. Tên hệ
  thống được ẩn và thứ tự được đảo độc lập theo mẫu.
- Ưu tiên JSON Schema, sau đó JSON Object, cuối cùng prompt-only JSON. Chế độ
  tương thích được xác định trên validation và khóa trước khi chạy test.

Căn cứ:
Thiết kế bảo đảm không chọn mẫu theo kết quả, giảm thiên lệch so sánh giữa các
candidate và cho phép tái lập dù chỉ có một human rater.

Ảnh hưởng:
Không thay manifest, schema prediction hoặc cấu hình của bốn hệ thống.

## 2026-08-10 — Pilot prompt v2 sau lỗi schema

Người phụ trách: Thành viên 4

Vấn đề:
Pilot v1 chỉ đạt 29/50 record hợp lệ. Prompt candidate chưa nêu trực tiếp tập
nhãn `error_types`, khiến provider ở chế độ JSON Object sinh nhãn tự do. Pilot
cũng dừng một phần do HTTP 402 `Insufficient Balance`.

Quyết định:

- Tăng `prompt_version` từ v1 lên v2; chạy lại đủ 50 request trên validation.
- Nêu chính xác 10 nhãn `error_types` trong cả system và user prompt.
- Parser chỉ chuẩn hóa biểu diễn chữ hoa/dấu cách/dấu gạch, vẫn từ chối nhãn khác
  nghĩa; mảng rỗng được quy về `none` hoặc `other` theo `major_error`.
- Giữ nguyên record v1 cho audit nhưng assessment chỉ đọc đúng prompt/model hiện
  tại, tránh trộn hai phiên bản.
- Lưu raw content của các lần validation thất bại trong `validation_failures`.
- HTTP 402 không retry. Nếu đổi provider/model, dùng `pilot --execute
  --renegotiate`; không dùng kết quả v1 để chạy main.

Ảnh hưởng:
Không thay sample, blind mapping, prediction hoặc rubric điểm 1–5. Đây là lần sửa
prompt thứ hai và cuối theo kế hoạch pilot.

## 2026-08-10 — Hoàn thiện prompt v2 và phân tích thống kê trước khi chạy lại

Người phụ trách: Thành viên 4

Bối cảnh:
Prompt v2 chưa được gọi API do pilot v1 hết số dư. Vì vậy các cải tiến dựa trên
G-Eval, SummEval, QAFactEval và nghiên cứu position bias được gộp vào cùng phiên
bản v2, không tạo prompt version thứ ba và không dùng kết quả test để tuning.

Quyết định:

- Thêm neo 1–5 cho từng tiêu chí; phân biệt omission với factual error.
- JSON-encode source/reference/candidate và coi toàn bộ giá trị là dữ liệu không
  đáng tin cậy để giảm prompt injection qua delimiter.
- Lấy 10 candidate pilot lặp theo tầng 3/3/2/2 thay vì mười record đầu.
- Chỉ retry lỗi provider/network và output model sai schema; lỗi cấu hình dừng ngay.
- Thêm paired bootstrap theo sample, xác suất A > B và tương quan ROUGE–LLM dạng
  chẩn đoán; giữ bootstrap marginal để báo CI từng hệ thống.
- Ghi latency, tổng token usage và fsync từng JSONL record.

Ảnh hưởng:
Phải chạy lại đủ pilot v2 sau khi giải quyết HTTP 402. Pilot v1 vẫn giữ nguyên
để audit tại thời điểm quyết định và không đủ điều kiện cho main run. Các
artefact v1 sau đó đã được xóa theo yêu cầu trước khi chạy pilot v2 sạch.

## 2026-08-10 — Khóa cấu hình sau pilot v2 và cho phép reference main

Người phụ trách: Thành viên 4

Bằng chứng:
Pilot v2 với `gemini/gemma-4-31b-it` đạt 50/50 record hợp lệ, first-pass valid
rate 100%, 40/40 so sánh điểm lặp lệch không quá 1 và độ lệch tối đa bằng 1.
Provider hỗ trợ strict `json_schema` cùng `max_completion_tokens`.

Quyết định:

- Khóa model, provider endpoint, prompt v2, response mode và token parameter
  trong `outputs/llm_judge/resolved_config.json`.
- Cho phép chạy reference audit chính thức; kết quả đạt 200/200 record hợp lệ.
- Không sửa prompt/rubric/sampling theo kết quả reference hoặc test.
- Không chạy candidate ba hệ thống dưới nhãn kết quả cuối và không tạo ViT5 giả.
- Khi có `outputs/predictions/vit5.jsonl`, chỉ chạy dry-run xác nhận đúng 800
  request rồi mới dùng `--execute` với resolved config hiện tại.

Ảnh hưởng:
Reference audit đã hoàn tất. Candidate scoring và human evaluation tiếp tục bị
khóa chỉ bởi thiếu ViT5; đây là blocker dữ liệu, không phải lỗi pipeline/API.
