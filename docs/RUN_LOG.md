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

## 2026-08-09–10 — Scratch Transformer corrected decoding sweep

### Phạm vi

- Manifest chính: `data/validation_select.jsonl`
- Số mẫu mỗi cấu hình: 2.000
- Checkpoint: `best_val_loss.pt`
- Tokenizer mô hình: `vietnamese_spm.model`
- Batch size: 2
- Device/precision: CPU/FP32
- Seed: 2026
- Cả ba cấu hình mới đã qua `test_smoke_10` trước khi chạy validation.

### Kết quả inference và validator

| Config ID | Thời gian | Success | Failed | Missing | Extra | Duplicate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `greedy` | 00:23:22 | 2000 | 0 | 0 | 0 | 0 |
| `beam4_lp0.8_nr3` | 04:06:59 | 2000 | 0 | 0 | 0 | 0 |
| `beam4_lp1.0_nr3` | 03:18:16 | 2000 | 0 | 0 | 0 | 0 |

Tất cả validator trả exit code 0 và `valid_for_handoff=true`.

SHA-256 prediction validation:

- `greedy`: `a2f96331dd084797881d066c0d1152939d805c2ec74d34b7f2daf406a081a7e5`
- `beam4_lp0.8_nr3`: `f235f95b7bd4219a163f85aaeed958b6523a10bad100d0b970eab94fb607bd8b`
- `beam4_lp1.0_nr3`: `2568c41ca97e6499506e6fbe039214e6193df7b7dfaf508b3b02dc982cf4fd67`

### Đánh giá tự động trên validation

Evaluator: `src/evaluate.py`; word segmentation: `underthesea==9.5.0`;
Unicode normalization: NFC.

| Config ID | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Compression (%) |
| --- | ---: | ---: | ---: | ---: |
| `greedy` | 33.0618 | 16.9446 | 27.5080 | 8.7365 |
| `beam4_lp0.8_nr3` | 33.7884 | 18.2143 | **28.4245** | 8.4232 |
| `beam4_lp1.0_nr3` | 33.9924 | 18.1839 | 28.3486 | 9.2057 |
| `beam4_lp1.1_nr3` | **34.1597** | **18.2425** | 28.3899 | 9.7061 |

Nhận xét: cả ba cấu hình beam vượt Greedy. LP=1.1 cao nhất ở ROUGE-1 và
ROUGE-2; LP=0.8 cao nhất ở ROUGE-L nhưng chỉ hơn LP=1.1 0,0346 điểm. Chưa tự
động đổi cấu hình test vì không có một cấu hình thắng tuyệt đối mọi metric;
quyết định cuối chờ nhóm xác nhận và không dùng test để lựa chọn.

Artifacts:

- `outputs/metrics/validation_scratch_greedy.json`
- `outputs/metrics/validation_scratch_beam4_lp0.8_nr3.json`
- `outputs/metrics/validation_scratch_beam4_lp1.0_nr3.json`
- `outputs/metrics/validation_scratch_beam4_lp1.1_nr3.json`
- `outputs/metrics/validation_scratch_decoding_comparison.json`
- `outputs/metrics/validation_scratch_decoding_comparison.csv`

