# Run Log

## 2026-08-06: Human Check Subsets Generation

### Overview
Generated two 10-sample subsets for human inspection:
- `data/validation_select_human_check.jsonl`
- `data/test_core_2000_human_check.jsonl`

### Sampling Parameters & Methodology
- **Random Seed**: `42` (`random.seed(42)`)
- **Total Samples Per File**: 10

#### Selection Priority Rules:
1. **Warning Samples**: All rows triggering row-level data quality warnings (e.g. `CONTROL_CHARACTER`, `OUTER_WHITESPACE`, `NON_NFC_UNICODE`).
2. **Min `source_words` Sample**: The sample with the shortest source text (by word count).
3. **Max `source_words` Sample**: The sample with the longest source text (by word count).
4. **Random Samples**: Random sampling without replacement from the remaining rows using `seed=42` until reaching 10 samples total per subset.

### Artifacts Created
- `src/create_human_check_subsets.py`
- `outputs/data_checks/human_check_sampling_manifest.json`
- `data/validation_select_human_check.jsonl`
- `data/test_core_2000_human_check.jsonl`

## 2026-08-06: Data Normalization

### Overview
Normalized data across 3 datasets (`validation_select.jsonl`, `test_smoke_10.jsonl`, `test_core_2000.jsonl`) using `clean_text`:
- Removed control character `\u0008` (backspace `\b`).
- Normalized Unicode text to NFC format (`unicodedata.normalize("NFC", text)`).

### Results
- `validation_select.jsonl`: 1 row modified (removed `\u0008`), 0 errors, 0 warnings.
- `test_smoke_10.jsonl`: 0 rows modified, 0 errors, 0 warnings.
- `test_core_2000.jsonl`: 1 row modified (removed `\u0008`), 0 errors, 0 warnings.

Validation script (`src/validate_data.py`) re-executed and passed cleanly with **0 errors and 0 warnings**.

## 2026-08-06: SHA-256 Checksum Generation

### Overview
Generated SHA-256 checksums for datasets after cleaning.

### Checksums
- `validation_select.jsonl`: `f7b088f59edf5538ff6eb5343e51dbfc6e148f86f02a562edb4058e3d41475aa`
- `test_smoke_10.jsonl`: `8f1327ac5b6df3881b5562f0aa4f5e0fe1254eb0d88a672e9947e9541910535c`
- `test_core_2000.jsonl`: `79a7aac04bc7aa708442bf0112391dae89f37c4432dc5f53dbb45b8ab0664da3`

Saved to `outputs/data_checks/sha256_checksums.json`.

### Verification Artifacts
- Created `data/MANIFEST_SHA256.txt` with expected checksums.
## 2026-08-07: Lead-1 & Lead-3 Baseline Generation

### Overview
Generated Lead-1 and Lead-3 baseline predictions for:
1. `data/test_smoke_10.jsonl` (10 samples)
2. `data/test_core_2000.jsonl` (2,000 samples)
3. `data/validation_select.jsonl` (2,000 samples)

### Sentence Segmentation Strategy
Implemented rule-based regex sentence splitter in `src/baselines.py` with:
- Web domain protection (`laodong.vn`, `google.com`).
- Date & number protection (`1.10.2019`, `1.234,5`).
- Title prefix protection (`BS.`, `TS.`, `GS.`, `PGS.`, `TP.`).
- Ellipsis handling (`...` followed by lowercase retained; followed by uppercase split).
- Quote-boundary handling (`.”` / `."` followed by uppercase split).

### Artifacts Created
- `src/baselines.py`
- `outputs/predictions/test/lead1.jsonl` (2,000 records)
- `outputs/predictions/test/lead3.jsonl` (2,000 records)
- `outputs/predictions/validation/lead1.jsonl` (2,000 records)
- `outputs/predictions/validation/lead3.jsonl` (2,000 records)
- `outputs/predictions/test_smoke_10/lead1.jsonl` (10 records)
- `outputs/predictions/test_smoke_10/lead3.jsonl` (10 records)

### Schema Validation
All generated files were validated against `configs/prediction_schema.json` using `jsonschema`: **100% Passed**.

## 2026-08-07: Evaluation Pipeline & Metric Calculation

### Overview
Developed evaluation pipeline in `src/evaluate.py` to calculate ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1, and Compression Ratio (%) based on prediction and manifest files linked by `id`.

### Key Specifications
- **Unicode Normalization**: NFC standard (`unicodedata.normalize("NFC", text)`), lowercase, whitespace/control character cleaning.
- **Vietnamese Tokenization**: Word segmentation via `underthesea.word_tokenize(..., format="text")` to preserve compound words (e.g. `thành_phố`).
- **Metrics**: Precision, Recall, F1 for ROUGE-1, ROUGE-2, ROUGE-L (LCS), and Compression Ratio ($\frac{\text{word\_count}(P)}{\text{word\_count}(S)} \times 100\%$).

### Artifacts Created
- `src/evaluate.py`
- `outputs/metrics/test_smoke_10_eval.json`
- `outputs/metrics/test_eval.json`
- `outputs/metrics/validation_eval.json`



## 2026-08-07 — Scratch Transformer smoke test

- Người chạy: member_2
- Input: `data/test_smoke_10.jsonl`
- Checkpoint: `best_val_loss.pt`
- Tokenizer: `vietnamese_spm.model`
- Config ID: `beam4_lp1.1_nr3`
- Batch size: 2
- Device/precision: CPU/FP32
- Input: 10
- Success: 10
- Failed: 0
- Output: `outputs/predictions/scratch_transformer_smoke_10.jsonl`
- Validator: exit code 0; missing=0; extra=0; duplicate=0
- Resume test: 10 record thành công được bỏ qua; output vẫn hợp lệ.

## 2026-08-07 — Scratch Transformer validation

- Người chạy: member_2
- Input: `data/validation_select.jsonl`
- Checkpoint: `best_val_loss.pt`
- Tokenizer: `vietnamese_spm.model`
- Config ID: `beam4_lp1.1_nr3`
- Batch size: 2
- Device/precision: CPU/FP32
- Input: 2000
- Success: 2000
- Failed: 0
- Output: `outputs/predictions/scratch_transformer_validation.jsonl`
- Validator: exit code 0; missing=0; extra=0; duplicate=0
- Ghi chú: Cấu hình được đóng băng trước khi chạy test core.

## 2026-08-07 — Scratch Transformer test core

- Người chạy: member_2
- Input: `data/test_core_2000.jsonl`
- Checkpoint: `best_val_loss.pt`
- Tokenizer: `vietnamese_spm.model`
- Config ID: `beam4_lp1.1_nr3`
- Batch size: 2
- Device/precision: CPU/FP32
- Thời gian: 03:22:25
- Input: 2000
- Success: 2000
- Failed: 0
- Output: `outputs/predictions/scratch_transformer.jsonl`
- Output SHA-256: `a4c0199d207c5fe4cd9323ae1f1cb67be150289349482a60e81cef149aece8ce`
- Validator: exit code 0; missing=0; extra=0; duplicate=0
- `valid_for_handoff=true`
- Không điều chỉnh decoding config dựa trên kết quả test.

## 2026-08-09 — Chuẩn bị deep-evaluation samples

- Người chạy: member_4
- Input: `data/test_core_2000.jsonl`, `data/validation_select.jsonl`
- Seed: 2026
- Phương pháp: bốn tầng bằng nhau theo hạng `(source_words, id)`
- LLM sample: 200, 50 mẫu/tầng
- Backup sample: 100, 25 mẫu/tầng và là subset của sample 200
- Pilot validation: 20, 5 mẫu/tầng
- Human sample: 50, phân bổ 13/13/12/12
- Output: `data/llm_judge_sample_200.jsonl`,
  `data/llm_judge_sample_100.jsonl`, `data/llm_judge_pilot_20.jsonl`,
  `data/human_sample.jsonl`, `outputs/llm_judge/sampling_manifest.json`
- API calls: chưa chạy; chỉ chuẩn bị dữ liệu và dry-run
- Blocker: chưa có `outputs/predictions/vit5.jsonl`, vì vậy candidate run và
  human evaluation cuối chưa sẵn sàng.

## 2026-08-10 — LLM judge pilot v1

- Người chạy: member_4
- Provider endpoint: `https://api.deepseek.com/...` (đã che)
- Model: `deepseek-v4-flash`
- Response mode: `json_object`
- Token parameter: `max_completion_tokens`
- Requested: 50; resume-skipped: 1; success mới: 28; failed: 21
- Tổng valid v1: 29/50; first-pass valid rate: 0.36
- Repeat score comparisons: 4; within-one rate: 0.50; max difference: 3
- Kết luận: `needs_review`, không đủ điều kiện chạy main.
- Phân tích lỗi: 17 lỗi validation output, chủ yếu `error_types` ngoài enum;
  4 request cuối bị HTTP 402 `Insufficient Balance`.
- Khắc phục: chuyển prompt sang v2 để liệt kê enum rõ ràng, tăng khả năng audit
  response sai schema và tách assessment theo prompt version/model.
- Blocker ngoài pipeline: cần nạp số dư provider hoặc đổi provider/model rồi chạy
  lại toàn bộ pilot v2.

## 2026-08-10 — Research-driven hardening trước pilot v2

- Nguồn đối chiếu: SummEval, G-Eval, QuestEval, QAFactEval, nghiên cứu position
  bias và tài liệu Structured Outputs chính thức.
- Thay đổi: neo điểm 1–5, JSON data envelope, fail-fast lỗi cấu hình, lặp pilot
  phân tầng 3/3/2/2, latency/token aggregation, paired bootstrap, ROUGE–LLM
  diagnostic.
- Kiểm thử: 28/28 unit/integration test đạt.
- API calls: 0; không phát sinh chi phí trong bước hardening.
- Prompt version: vẫn là v2 vì v2 chưa được execute; v1 giữ nguyên để audit.
- Blocker còn lại: HTTP 402 của provider và thiếu `outputs/predictions/vit5.jsonl`.

## 2026-08-10 — LLM judge pilot v2 đạt ngưỡng

- Người chạy: member_4
- Model: `gemini/gemma-4-31b-it`
- Provider endpoint: `http://host-49960de5880e:20128/...` (hostname đã băm)
- Prompt version: `v2`
- Response mode: `json_schema`
- Token parameter: `max_completion_tokens`
- Requested: 50; capability-probe/resume skipped: 1; success mới: 49; failed: 0
- Tổng valid v2: 50/50; first-pass valid rate: 1.00
- Repeat score comparisons: 40; within-one rate: 1.00; max difference: 1
- Token usage: prompt 204.902; completion 7.140; total 212.042
- Kết luận: `passed`; đạt toàn bộ ngưỡng pilot và khóa model/provider/prompt.
- Output: `outputs/llm_judge/pilot_raw.jsonl`, `pilot_scores.jsonl`,
  `pilot_run.json`, `resolved_config.json`.
- Pilot v1 đã được xóa theo yêu cầu trước khi chạy v2; không trộn record v1/v2.

## 2026-08-10 — Reference audit chính thức

- Người chạy: member_4
- Input: `data/llm_judge_sample_200.jsonl`
- Model/provider/prompt: giữ nguyên resolved config đã khóa từ pilot v2
- Requested: 200; skipped: 0; success: 200; failed: 0
- Unique valid records: 200/200
- Token usage: prompt 790.906; completion 28.698; total 819.604
- Tổng latency cộng dồn: 5.377,3217 giây; đây không phải wall-clock do concurrency 5.
- Output: `outputs/llm_judge/reference_raw.jsonl`, `reference_scores.jsonl`,
  `reference_run.json`.
- Candidate run chưa phát sinh request vì pipeline dừng trước API khi thiếu
  `outputs/predictions/vit5.jsonl`.
- Blocker hiện tại duy nhất của candidate/human evaluation: chưa có ViT5 test prediction.

