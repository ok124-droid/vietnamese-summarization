# Scratch Transformer inference

Pipeline này chỉ phục vụ suy luận cho checkpoint Scratch Transformer đã cung cấp.
Không huấn luyện lại model/tokenizer và không dùng `test_core_2000` để chỉnh decoding.

## Tài sản cố định

- Model: Transformer encoder–decoder tự xây dựng, Pre-LN.
- Config: `configs/transformer_summarization.yaml`.
- Checkpoint: `checkpoints/transformer_base/best_val_loss.pt`.
- Tokenizer: SentencePiece Unigram 16.000 từ tại
  `data/tokenizer/vietnamese_spm.model`.
- Checksum và ghi chú provenance: `checkpoint_info.txt`.

Pipeline đọc manifest có `id` và `source`. Prediction cuối chỉ có đúng:

```json
{
  "id": "...",
  "system": "scratch_transformer",
  "config_id": "...",
  "prediction": "...",
  "status": "ok",
  "error": null
}
```

## Cài đặt

Từ thư mục gốc repository trên PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r scratch_transformer_inference\requirements.txt
```

Nếu dùng GPU, cài đúng bản PyTorch phù hợp CUDA của máy trước, sau đó cài các
dependency còn lại. Kiểm tra bằng:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Một mẫu để kiểm tra nhanh

```powershell
.\.venv\Scripts\python.exe scratch_transformer_inference\src\infer.py `
  --input data\test_smoke_10.jsonl `
  --output outputs\predictions\scratch_transformer_one.jsonl `
  --checkpoint scratch_transformer_inference\checkpoints\transformer_base\best_val_loss.pt `
  --tokenizer scratch_transformer_inference\data\tokenizer\vietnamese_spm.model `
  --config scratch_transformer_inference\configs\transformer_summarization.yaml `
  --decoding-config scratch_transformer_inference\configs\decoding_beam4_lp1.1_nr3.json `
  --batch-size 1 --device cpu --precision fp32 --limit 1 --overwrite
```

## Smoke test 10 mẫu

```powershell
.\.venv\Scripts\python.exe scratch_transformer_inference\src\infer.py `
  --input data\test_smoke_10.jsonl `
  --output outputs\predictions\scratch_transformer_smoke_10.jsonl `
  --checkpoint scratch_transformer_inference\checkpoints\transformer_base\best_val_loss.pt `
  --tokenizer scratch_transformer_inference\data\tokenizer\vietnamese_spm.model `
  --config scratch_transformer_inference\configs\transformer_summarization.yaml `
  --decoding-config scratch_transformer_inference\configs\decoding_beam4_lp1.1_nr3.json `
  --batch-size 1 --device auto --precision fp32 --overwrite
```

Validator smoke:

```powershell
.\.venv\Scripts\python.exe scratch_transformer_inference\src\validate_predictions.py `
  --manifest data\test_smoke_10.jsonl `
  --predictions outputs\predictions\scratch_transformer_smoke_10.jsonl `
  --expected-config-id beam4_lp1.1_nr3
```

Chỉ chuyển sang validation khi validator trả exit code 0 và đã đọc thủ công ít
nhất ba prediction.

## Chạy validation để xác nhận decoding

```powershell
.\.venv\Scripts\python.exe scratch_transformer_inference\src\infer.py `
  --input data\validation_select.jsonl `
  --output outputs\predictions\scratch_transformer_validation.jsonl `
  --checkpoint scratch_transformer_inference\checkpoints\transformer_base\best_val_loss.pt `
  --tokenizer scratch_transformer_inference\data\tokenizer\vietnamese_spm.model `
  --config scratch_transformer_inference\configs\transformer_summarization.yaml `
  --decoding-config scratch_transformer_inference\configs\decoding_beam4_lp1.1_nr3.json `
  --batch-size 4 --device cuda --precision fp32 --overwrite
```

Đánh giá cấu hình chỉ trên validation. Sau khi chốt, ghi quyết định vào
`docs/DECISIONS.md` và không đổi cấu hình dựa trên test.

## Test core 2.000 mẫu

Chỉ chạy sau khi smoke đạt và decoding đã được đóng băng:

```powershell
.\.venv\Scripts\python.exe scratch_transformer_inference\src\infer.py `
  --input data\test_core_2000.jsonl `
  --output outputs\predictions\scratch_transformer.jsonl `
  --checkpoint scratch_transformer_inference\checkpoints\transformer_base\best_val_loss.pt `
  --tokenizer scratch_transformer_inference\data\tokenizer\vietnamese_spm.model `
  --config scratch_transformer_inference\configs\transformer_summarization.yaml `
  --decoding-config scratch_transformer_inference\configs\decoding_beam4_lp1.1_nr3.json `
  --batch-size 4 --device cuda --precision fp32 --overwrite
```

Resume dùng lại chính lệnh trên nhưng thay `--overwrite` bằng `--resume`.
`--resume` giữ các ID `status="ok"` và tự chạy lại ID lỗi. Không cần thêm
`--retry-errors`; cờ này chỉ còn để tương thích CLI cũ.

Validator test core:

```powershell
.\.venv\Scripts\python.exe scratch_transformer_inference\src\validate_predictions.py `
  --manifest data\test_core_2000.jsonl `
  --predictions outputs\predictions\scratch_transformer.jsonl `
  --expected-config-id beam4_lp1.1_nr3
```

Validator trả exit code khác 0 nếu JSON/schema/ID/config sai hoặc còn bất kỳ
record lỗi nào.

## Precision, batch và nhiều GPU

- Mặc định là FP32. `training.use_amp` trong config huấn luyện không tự bật AMP.
- Chỉ dùng `--precision fp16` hoặc `bf16` sau khi đã kiểm tra smoke output.
- Batch size là quyết định vận hành; giảm về 1 khi OOM.
- `torch.compile` chỉ bật khi truyền `--compile`.
- DDP chỉ bật khi chạy bằng `torchrun`; ví dụ hai GPU:

```powershell
torchrun --standalone --nproc_per_node=2 scratch_transformer_inference\src\infer.py <các tham số ở trên>
```

## Sau mỗi lần chạy chính

1. Chạy validator và lưu exit code.
2. Cập nhật `outputs/predictions/scratch_transformer.meta.json` bằng thông số thật.
3. Ghi lệnh, số success/failed và output vào `docs/RUN_LOG.md`.
4. Không xóa mẫu xấu và không đưa `source`/`reference` vào prediction.
5. Không chạy test core lại với decoding khác chỉ vì đã xem kết quả test.

## Chạy bổ sung ba cấu hình decoding

Theo yêu cầu đánh giá bổ sung, ba cấu hình cố định được lưu tại:

- `configs/decoding_greedy.json`
- `configs/decoding_beam4_lp0.8_nr3.json`
- `configs/decoding_beam4_lp1.0_nr3.json`

Greedy ở đây là greedy tiêu chuẩn (`num_beams=1`,
`no_repeat_ngram_size=0`). Hai cấu hình beam cùng dùng beam size 4 và
no-repeat n-gram 3; chỉ thay length penalty.

Từ thư mục gốc repository, chạy smoke và tự động validate cả ba cấu hình:

```powershell
.\scratch_transformer_inference\scripts\run_decoding_sweep.ps1 `
  -Split smoke `
  -Mode overwrite `
  -BatchSize 2 `
  -Device cpu `
  -Precision fp32
```

Sau khi cả ba smoke output đạt validator và đã kiểm tra thủ công, chạy trên
validation:

```powershell
.\scratch_transformer_inference\scripts\run_decoding_sweep.ps1 `
  -Split validation `
  -Mode overwrite `
  -BatchSize 2 `
  -Device cpu `
  -Precision fp32
```

Nếu job bị ngắt, chạy lại cùng lệnh nhưng dùng `-Mode resume`. Các output được
ghi riêng dưới `outputs/predictions/test_smoke_10/` hoặc
`outputs/predictions/validation/`; script không ghi đè prediction test core đã
bàn giao. Chỉ dùng kết quả validation để so sánh cấu hình.
