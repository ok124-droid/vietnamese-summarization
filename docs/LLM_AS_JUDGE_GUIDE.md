# Hướng dẫn LLM-as-a-Judge và human evaluation

## Phạm vi

Pipeline đánh giá sâu 200 mẫu cố định từ `test_core_2000`:

- 200 lượt reference audit.
- 800 lượt candidate scoring, bốn hệ thống và một candidate mỗi request.
- Human evaluation 50 mẫu với một người chấm.
- 10 sample block lặp ẩn để đo độ ổn định nội bộ của người chấm.

Không dùng test để chọn prompt, model hoặc cấu hình. Pilot chỉ chạy trên 20 mẫu
`validation_select` đã cố định.

## Cài đặt và cấu hình API

```powershell
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "model-id-cua-ban"
```

Không ghi API key vào config, log hoặc Git. `OPENAI_BASE_URL` có thể trỏ đến
provider tương thích OpenAI Chat Completions. Model và base URL không được đổi
giữa pilot và lần chạy chính.

Pilot thử lần lượt:

1. `response_format=json_schema`.
2. `response_format=json_object` nếu provider báo không hỗ trợ tham số.
3. Prompt-only JSON nếu cả hai chế độ trên không được hỗ trợ.

Chế độ thành công được lưu trong `outputs/llm_judge/resolved_config.json`.
Nếu provider trả confidence theo xác suất 0-1 hoặc chuỗi số, parser chuẩn hóa về
thang 1-5; raw response vẫn được giữ nguyên để kiểm tra. JSON hợp lệ nhưng sai
schema được retry tối đa theo `max_retries` thay vì làm dừng toàn bộ lần chạy.
Các response sai schema cũng được ghi vào `validation_failures` trong raw/error
JSONL để chẩn đoán. Prompt candidate v2 liệt kê đầy đủ enum `error_types`, neo
riêng cho từng mức 1–5 và yêu cầu đối chiếu khẳng định với source trước khi điền
form. Input được JSON-encode và coi là dữ liệu không đáng tin cậy. Parser chỉ
chuẩn hóa khác biệt về chữ hoa, dấu cách và dấu gạch, không chấp nhận nhãn khác
nghĩa ngoài rubric. Lỗi cấu hình/invariant không được retry.

## Quy trình chạy

### 1. Tạo mẫu và mapping ẩn danh

```powershell
python src/run_llm_judge.py prepare
```

Lệnh tạo sample 200/100, pilot 20, human sample 50, sampling manifest, SHA-256
và blind mapping. Sampling dùng `seed=2026`, bốn tầng bằng nhau theo hạng độ
dài source.

### 2. Kiểm tra số request trước khi gọi API

```powershell
python src/run_llm_judge.py pilot
python src/run_llm_judge.py run --task reference
python src/run_llm_judge.py run --task candidate
```

Ba lệnh trên là dry-run. Candidate dry-run chỉ đạt 800 request khi đủ bốn file
prediction, bao gồm `outputs/predictions/vit5.jsonl`. Dry-run cũng báo ước lượng
input token dựa trên số ký tự; đây không phải token usage tính phí thực tế.

### 3. Pilot có phát sinh chi phí

```powershell
python src/run_llm_judge.py pilot --execute
```

Pilot gồm 20 reference, 20 candidate và 10 candidate lặp phân tầng 3/3/2/2 theo
độ dài source. Kiểm tra thủ công file
parsed/raw, khóa prompt version và xác nhận mức ổn định trước khi chạy test.
Khi prompt version thay đổi, pipeline chạy lại đủ 50 request và chỉ đánh giá các
record có cùng prompt version/model; record cũ vẫn được giữ để audit nhưng không
trộn vào kết quả mới.

Nếu cố ý đổi provider hoặc model sau lỗi quota/số dư, đặt lại các biến môi trường
rồi yêu cầu dò lại capability:

```powershell
python src/run_llm_judge.py pilot --execute --renegotiate
```

HTTP 402 là lỗi tài khoản/provider và không được retry. Cần nạp thêm số dư hoặc
đổi provider/model trước khi tiếp tục; không chạy main khi pilot mới chưa đạt ngưỡng.

### 4. Chạy chính có resume

```powershell
python src/run_llm_judge.py run --task reference --execute
python src/run_llm_judge.py run --task candidate --execute
```

Record hợp lệ đã có trong file parsed sẽ được bỏ qua theo khóa
`task|sample_id|anon_candidate_id|prompt_version|model|run_id`. Raw response,
parsed result, usage, latency, response ID và lỗi cuối cùng được lưu riêng. Run
metadata có tổng token usage/latency của mọi record hợp lệ cùng prompt/model.
Không xóa file đầu ra khi resume.

### 5. Tạo và chấm human evaluation

```powershell
python src/run_llm_judge.py prepare-human
```

Điền các cột điểm 1-5 và `major_error` trong
`outputs/human/human_evaluation_blind.csv`. Không mở
`human_blind_mapping.json` trước khi khóa file chấm. Có 60 block, 240 candidate
row; 10 block là bản lặp ẩn và cách bản gốc ít nhất 15 block.

### 6. Tổng hợp

```powershell
python src/run_llm_judge.py summarize
```

Lệnh chỉ chạy khi có đủ 200 reference score và 800 candidate score. Kết quả gồm:

- Tỉ lệ reference label và Wilson CI 95%.
- Điểm trung bình/trung vị, bootstrap CI 95%, major-error rate và error type.
- Paired bootstrap theo sample cho sáu cặp hệ thống và xác suất A > B.
- Spearman giữa ROUGE-L chẩn đoán và từng chiều LLM judge.
- Spearman, MAE, within-one giữa LLM và human.
- Quadratic weighted kappa trên các block human lặp.
- Bảy ví dụ định tính và bảng CSV dùng cho báo cáo.

## Rubric

### Reference audit

- `fully_supported`: mọi ý cốt lõi có căn cứ trong source.
- `partially_supported`: có cả ý được hỗ trợ và không được hỗ trợ.
- `unsupported`: phần lớn khẳng định không có căn cứ trong source.
- `mismatched`: reference nói về sự kiện hoặc bài báo khác.

### Candidate scoring

- Faithfulness: đúng với source, không bịa hoặc mâu thuẫn.
- Informativeness: bao phủ ý quan trọng của source.
- Fluency: tự nhiên, rõ ràng và đúng ngữ pháp.
- Conciseness: súc tích, không lặp hoặc chứa chi tiết thừa.

Mỗi tiêu chí có mô tả riêng cho cả năm mức trong prompt v2. Faithfulness và
informativeness chỉ dựa trên source; reference không được gửi trong candidate
request. Thiếu ý làm giảm informativeness nhưng không tự động làm giảm
faithfulness nếu các ý đã viết vẫn đúng.

Phân tích phương pháp và liên hệ với SummEval, G-Eval, QuestEval/QAFactEval cùng
position bias được ghi tại `docs/DEEP_EVALUATION_RESEARCH.md`.

## Blocker và nguyên tắc báo cáo

Nếu thiếu ViT5, hoàn tất sampling, pilot và reference audit nhưng chỉ xem kết quả
candidate ba hệ thống là provisional. Không tạo prediction giả và không trình bày
bảng ba hệ thống như kết quả cuối. Human evaluation chỉ được tạo sau khi đủ bốn
prediction.

Vì chỉ có một người chấm, báo cáo không được tuyên bố inter-rater agreement.
Quadratic weighted kappa từ 10 block lặp chỉ phản ánh intra-rater consistency.
