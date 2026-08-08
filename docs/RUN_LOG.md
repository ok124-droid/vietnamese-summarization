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

