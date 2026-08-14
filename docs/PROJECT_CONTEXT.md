# PROJECT CONTEXT

> Bối cảnh ổn định để thành viên và coding agent hiểu dự án mà không phải đọc lại toàn bộ repository.

## 1. Cách dùng trong chat mới

Agent cần:

1. Đọc file này.
2. Đọc phần bàn giao của người dùng: việc đã làm, mục tiêu mới và các file cần đọc.
3. Chỉ mở các file được chỉ định và dependency trực tiếp cần thiết.
4. Không quét toàn bộ repository, dữ liệu, checkpoint, prediction hoặc log cũ.
5. Nếu cần đọc thêm file, nêu lý do ngắn gọn trước.

Thứ tự ưu tiên khi có mâu thuẫn:

1. Yêu cầu mới nhất của người dùng.
2. Những quyết định đã được chốt trong RUN_LOG.md.
3. File này.
4. Tài liệu chuyên biệt.
5. Code hoặc log cũ chưa được xác nhận.

File này không lưu tiến độ. Không được suy ra một việc đã hoàn thành nếu phần bàn giao hoặc `RUN_LOG.md` chưa xác nhận.

---

## 2. Thông tin cốt lõi

| Mục | Nội dung |
|---|---|
| Dự án | Tìm hiểu và đánh giá bài toán tóm tắt văn bản tiếng Việt |
| Học phần | Xử lý ngôn ngữ tự nhiên — Summer 2026 |
| Nhóm | 4 thành viên |
| Hạn nộp | 24:00 ngày 11/08/2026 |
| Phạm vi | Tóm tắt **đơn văn bản** cho tin tức tiếng Việt |
| Dữ liệu | News Dataset Vietnamese, chủ yếu từ Báo Lao Động |
| Hệ thống bắt buộc | Lead-1, Lead-3, Scratch Transformer, ViT5 |
| Kết quả chính | So sánh bốn hệ thống trên cùng `test_core_2000` |

Scratch Transformer, tokenizer và checkpoint được kế thừa từ dự án trước. Đóng góp chính của dự án hiện tại là chuẩn hóa dữ liệu, so sánh công bằng, đánh giá đúng cho tiếng Việt, kiểm tra chất lượng reference và phân tích lỗi.

Scratch Transformer không cần tốt hơn ViT5. Giá trị của dự án nằm ở quy trình đánh giá minh bạch, có thể kiểm tra và chạy lại.

Không dùng ViMs/VietnameseMDS cho thực nghiệm chính vì chúng là dữ liệu tóm tắt đa văn bản. Không thêm hệ thống mở rộng hoặc huấn luyện lại mô hình trước khi phần đánh giá lõi hoàn thành.

---

## 3. Dữ liệu và manifest

Dữ liệu gốc:

- Kaggle: <https://www.kaggle.com/datasets/phamtheds/news-dataset-vietnameses>
- `Contents` → `source`
- `Summary` → `reference`

Manifest đã đóng băng là nguồn sự thật cho thực nghiệm hiện tại:

| File | Mục đích |
|---|---|
| `validation_select.jsonl` | Chọn cấu hình và pilot phương pháp đánh giá |
| `test_smoke_10.jsonl` | Kiểm tra pipeline trên đúng 10 mẫu |
| `test_core_2000.jsonl` | Tập test lõi gồm đúng 2.000 mẫu |
| `llm_judge_sample_200.jsonl` | Mẫu con từ test core để LLM đánh giá sâu |
| `human_sample.jsonl` | 30–50 mẫu con để con người chấm |

Manifest có thể nằm trong `data/` hoặc `data/manifests/`; phần bàn giao phải ghi đường dẫn thật.

Quan hệ bắt buộc:

```text
test_smoke_10 ⊂ test_core_2000
validation_select ∩ test_core_2000 = rỗng
```

- ID validation có dạng `validation_XXXXXX`; ID test có dạng `test_XXXXXX`.
- ID được gán trước khi lấy mẫu và không được đánh lại sau khi lấy subset.
- Cùng một ID trong smoke và core phải có nội dung giống hệt nhau.
- File `*_human_check.jsonl` chỉ là mẫu kiểm tra thủ công, không phải manifest đầy đủ để chạy mô hình.

Schema mỗi dòng manifest:

```json
{
  "id": "test_000127",
  "split": "test",
  "source": "Nội dung bài báo...",
  "reference": "Tóm tắt tham chiếu..."
}
```

Mỗi dòng phải là JSON UTF-8 có đúng bốn trường chuỗi, không trùng ID và không có trường rỗng.

Làm sạch kỹ thuật được phép:

- loại U+0008 khỏi `source` và `reference`;
- chuẩn hóa Unicode NFC trên tất cả split.

Không tự động sửa chính tả, số liệu, tên riêng, viết lại reference hoặc loại mẫu theo cảm nhận. Sau khi đóng băng, tạo SHA-256 cho ba manifest chính. Nếu manifest thay đổi, phải sinh lại mọi prediction và metric phụ thuộc.

Một số reference có thể là sapô, thiếu ý, chứa thông tin không có trong source hoặc không khớp. Vì vậy không coi reference hay ROUGE là chân lý tuyệt đối.

---

## 4. Các hệ thống và prediction

| Hệ thống | Quy ước |
|---|---|
| Lead-1 | Lấy câu đầu tiên của source |
| Lead-3 | Lấy ba câu đầu tiên; dùng cùng hàm tách câu với Lead-1 |
| Scratch Transformer | Transformer encoder–decoder khoảng 50M tham số; checkpoint, SentencePiece tokenizer và config phải đi cùng nhau |
| ViT5 | Ghi chính xác model ID; chỉ chọn cấu hình trên validation |

Cấu hình decoding tham chiếu của Scratch Transformer:

```text
beam_search; beam_size=4; length_penalty=1.1
no_repeat_ngram_size=3; config_id=beam4_lp1.1_nr3
```

Cấu hình cuối phải được xác nhận trên validation và đóng băng trước khi chạy test. Không huấn luyện tokenizer mới để dùng với checkpoint cũ.

Schema prediction:

```json
{
  "id": "test_000127",
  "system": "scratch_transformer",
  "config_id": "beam4_lp1.1_nr3",
  "prediction": "Tóm tắt do hệ thống sinh...",
  "status": "ok",
  "error": null
}
```

Quy tắc:

- Đúng một prediction cho mỗi ID trong manifest; không thiếu, trùng hoặc có ID lạ.
- `status` chỉ là `ok` hoặc `error`.
- `status="ok"` yêu cầu prediction không rỗng và `error=null`.
- Không lặp lại source/reference trong file prediction.
- Evaluator ghép bằng `id`, không ghép mù theo vị trí dòng.
- Không âm thầm bỏ mẫu lỗi khi tính metric.

---

## 5. Quy trình và đánh giá

Quy trình bắt buộc:

1. Kiểm tra và đóng băng manifest.
2. Smoke test từng hệ thống trên 10 mẫu.
3. Chọn decoding/model/prompt trên validation.
4. Đóng băng checkpoint, tokenizer, config và prompt.
5. Chạy bốn hệ thống trên cùng `test_core_2000`.
6. Kiểm tra schema và độ bao phủ ID.
7. Tính metric tự động.
8. Đánh giá bằng LLM, con người và phân tích lỗi.

Metric lõi:

- ROUGE-1, ROUGE-2, ROUGE-L với xử lý Unicode tiếng Việt đúng; F1 là kết quả chính;
- độ dài và tỉ lệ nén;
- lặp n-gram và novel n-gram;
- lead bias;
- bootstrap confidence interval 95% nếu đủ thời gian.

Mọi hệ thống phải dùng cùng manifest, evaluator và normalization. **Không dùng test để chọn cấu hình, model hoặc sửa prompt.**

LLM-as-a-Judge:

- pilot trên 20 mẫu validation;
- tập chính gồm 200 mẫu từ test core, chia bốn nhóm độ dài và lấy 50 mẫu mỗi nhóm bằng seed `2026`;
- nếu thiếu quota, dùng 100 mẫu đã xác định trước, 25 mẫu mỗi nhóm;
- ẩn tên bốn hệ thống thành A/B/C/D và lưu ánh xạ riêng;
- reference audit nhận `source + reference`;
- candidate evaluation chỉ nhận `source + bốn candidate`, không nhận reference;
- chấm Faithfulness, Informativeness, Fluency và Conciseness từ 1–5;
- ghi chính xác provider, model ID, prompt version, seed và cấu hình;
- không coi LLM judge là ground truth hoặc human evaluation.

Human evaluation: tối thiểu 30 mẫu, mục tiêu 50; candidate được ẩn tên và chấm bằng cùng rubric.

---

## 6. Quy tắc không được vi phạm

- Không sửa hoặc đổi manifest, ID, schema hay metric mà không báo nhóm.
- Không dùng test để tuning.
- Không loại mẫu khỏi đánh giá chính sau khi xem prediction.
- Không so sánh các hệ thống trên các tập ID khác nhau.
- Không dùng tokenizer không khớp checkpoint Scratch Transformer.
- Không dùng tokenizer ROUGE làm mất chữ tiếng Việt có dấu.
- Không tính metric trên subset có lợi cho một hệ thống mà không báo rõ.
- Không đưa API key, checkpoint hoặc dữ liệu lớn lên Git; dùng biến môi trường, link tải và checksum.
- Mỗi lần chạy chính phải lưu input/checksum, model/checkpoint, tokenizer, config, seed, lệnh chạy, output, số mẫu thành công/thất bại và lỗi đã biết.
- Quyết định quan trọng ghi vào `docs/DECISIONS.md`; các lần chạy ghi vào `docs/RUN_LOG.md`.

Phân công ổn định:

| Thành viên | Phụ trách |
|---|---|
| 1 | Dữ liệu, manifest, Lead-1/Lead-3, metric tự động |
| 2 | Scratch Transformer |
| 3 | ViT5 |
| 4 | LLM-as-a-Judge, reference audit và human evaluation |

---

## 7. Tài liệu cần biết

- `PROJECT_CONTEXT.md`: context ổn định, đọc đầu tiên.
- `docs/PROJECT_BRIEF.md`: mô tả chi tiết dự án.
- `docs/LLM_AS_JUDGE_GUIDE.md`: rubric và quy trình judge.
- `docs/DECISIONS.md`: quyết định mới.
- `docs/RUN_LOG.md`: tiến độ, lệnh chạy, kết quả và lỗi.

Chỉ mở tài liệu chi tiết khi nhiệm vụ hiện tại thực sự cần.

---

## 8. Mẫu bàn giao cho chat mới

```markdown
## BÀN GIAO

### Đã hoàn thành
- ...

### Mục tiêu của chat này
- ...

### File bắt buộc phải đọc
- `path/to/file`: lý do

### Input và output mong muốn
- Input: ...
- Output: ...

### Lỗi hoặc điều đã thử
- ...

### Điều kiện hoàn thành
- ...
```

Prompt mở đầu:

```text
Đọc PROJECT_CONTEXT.md và phần bàn giao trước. Chỉ đọc các file được liệt kê và dependency trực tiếp cần thiết. Không quét toàn bộ repository. Nếu cần mở thêm file, hãy nêu lý do trước.
```

Chỉ cập nhật file này khi thay đổi phạm vi, manifest, schema, hệ thống bắt buộc, quy trình đánh giá hoặc phân công ổn định. Tiến độ và lỗi tạm thời không thuộc file này.
