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

## 2026-08-08: Scratch Transformer Validation Evaluation

### Overview
Executed evaluation script `src/evaluate.py` to calculate ROUGE and Compression Ratio metrics for `outputs/predictions/validation/scratch_transformer_beam4_lp1.1_nr3.jsonl` (formerly `scratch_transformer_validation.jsonl`) on `data/validation_select.jsonl` (2,000 samples).

### Results Summary
- **ROUGE-1 F1**: `34.1597`
- **ROUGE-2 F1**: `18.2425`
- **ROUGE-L F1**: `28.3899`
- **Compression Ratio (%)**: `9.7061`
- **Valid Predictions**: `2,000 / 2,000` (`0` errors)

### Artifact Updated
- `outputs/metrics/validation_eval.json`

## 2026-08-10: ViT5 Validation Evaluation (Hyperparameter Tuning)

### Overview
Evaluated ViT5 model predictions with different decoding length penalties (`lp=0.8`, `lp=1.0`, `lp=1.1`) on `data/validation_select.jsonl` (2,000 samples).

### Results Summary
- **validation_predictions_beam4_lp0.8_max128_nr3.jsonl**:
  - ROUGE-1 F1: `31.1089` | ROUGE-2 F1: `14.1935` | ROUGE-L F1: `24.6530` | Comp Ratio: `8.7852%`
- **validation_predictions_beam4_lp1_max128_nr3.jsonl**:
  - ROUGE-1 F1: `31.2016` | ROUGE-2 F1: `14.2340` | ROUGE-L F1: `24.6987` | Comp Ratio: `8.8863%`
- **validation_predictions_beam4_lp1.1_max128_nr3.jsonl**:
  - ROUGE-1 F1: `31.2495` | ROUGE-2 F1: `14.2283` | ROUGE-L F1: `24.7108` | Comp Ratio: `8.9242%`

### Artifact Updated
- `outputs/metrics/validation_eval.json`

## 2026-08-11: Test Core 2.000 Evaluation

### Overview
Executed `src/evaluate.py` to evaluate all candidate systems on the frozen test set `data/test_core_2000.jsonl` (2,000 samples).

### Results Summary
- **Scratch Transformer** (`scratch_transformer_test_core_2000.jsonl`):
  - ROUGE-1 F1: `33.8438` | ROUGE-2 F1: `17.8399` | ROUGE-L F1: `28.3006` | Comp Ratio: `9.4176%`
- **ViT5 Base** (`vit5_test_core_2000.jsonl`):
  - ROUGE-1 F1: `31.2188` | ROUGE-2 F1: `14.5080` | ROUGE-L F1: `25.0001` | Comp Ratio: `8.6309%`
- **Lead-1 Baseline** (`lead1.jsonl`):
  - ROUGE-1 F1: `26.7782` | ROUGE-2 F1: `12.1660` | ROUGE-L F1: `20.8032` | Comp Ratio: `11.5062%`
- **Lead-3 Baseline** (`lead3.jsonl`):
  - ROUGE-1 F1: `25.1245` | ROUGE-2 F1: `11.0348` | ROUGE-L F1: `17.9164` | Comp Ratio: `30.1437%`

### Artifact Updated
- `outputs/metrics/test_eval.json`






