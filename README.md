# Vietnamese News Summarization

> Xây dựng và đánh giá hệ thống tóm tắt đơn văn bản cho tin tức tiếng Việt, với bốn cấu hình: **Lead-1**, **Lead-3**, **Transformer huấn luyện từ đầu** và **ViT5**.

Repository tập trung không chỉ vào ROUGE mà còn vào khả năng tái lập, chất lượng dữ liệu, độ dài đầu ra và đánh giá source-grounded bằng LLM.

## Kết quả chính

### Đánh giá tự động trên `test_core_2000`

| Hệ thống | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | CR (%) |
|---|---:|---:|---:|---:|
| Lead-1 | 26.78 | 12.17 | 20.80 | 11.51 |
| Lead-3 | 25.12 | 11.03 | 17.92 | 30.14 |
| **Scratch Transformer** | **33.84** | **17.84** | **28.30** | 9.42 |
| ViT5 | 31.22 | 14.51 | 25.00 | **8.63** |

Scratch Transformer đạt ROUGE cao nhất trên cả ba biến thể. Tuy nhiên, ROUGE không được xem là kết luận duy nhất vì reference của tập dữ liệu có chất lượng không đồng đều.

### LLM-as-a-Judge trên 200 source

Mỗi source được ghép với tóm tắt của cả bốn hệ thống. Tên hệ thống được ẩn, vị trí A/B/C/D được đảo và cân bằng; toàn bộ 800 candidate được đánh giá theo cùng rubric.

| Hệ thống | Faithfulness | Coverage | Focus | Fluency | Conciseness | Overall | Major factual error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lead-1 | 5.00 | 2.42 | 4.06 | **4.71** | 4.25 | 4.09 | **0.0%** |
| **Lead-3** | **5.00** | **3.42** | **4.15** | 4.63 | 4.00 | **4.32** | **0.0%** |
| Scratch Transformer | 3.39 | 2.16 | 3.62 | 4.10 | 4.47 | 3.33 | 45.5% |
| ViT5 | 3.26 | 2.22 | 3.81 | 4.34 | **4.63** | 3.38 | 52.0% |

Khoảng tin cậy bootstrap 95% của chênh lệch overall giữa Scratch Transformer và ViT5 chứa 0, vì vậy chưa có bằng chứng rằng một trong hai hệ thống sinh tốt hơn rõ ràng theo rubric hiện tại. Lead-3 có Coverage và overall cao nhất, nhưng CR trên cùng 200 mẫu là 29.00%, lớn hơn nhiều so với ba hệ thống còn lại; đây là trade-off giữa độ bao phủ và độ súc tích.

### Kiểm tra chất lượng reference

Reference audit trên 200 mẫu cho thấy:

- 54.0% `fully_supported`;
- 37.5% `partially_supported`;
- 8.5% `unsupported`;
- 0% `mismatched`.

Như vậy, 46.0% reference có ít nhất một ý cốt lõi không được source hỗ trợ. Vì thế, dự án báo cáo ROUGE để bảo đảm khả năng so sánh, đồng thời dùng đánh giá trực tiếp dựa trên source để phân tích factuality và chất lượng sử dụng thực tế.

## Mục tiêu dự án

- Chuẩn hóa dữ liệu tin tức tiếng Việt thành cùng schema và đóng băng manifest bằng SHA-256.
- Xây dựng baseline Lead-1 và Lead-3 với bộ tách câu có xử lý đặc thù tiếng Việt.
- Đóng gói Transformer encoder-decoder huấn luyện từ đầu cùng tokenizer, cấu hình và inference script.
- So sánh công bằng với `VietAI/vit5-base-vietnews-summarization`.
- Chọn cấu hình trên validation, không tuning trên test.
- Đánh giá đa chiều bằng ROUGE, compression ratio, kiểm tra reference và LLM-as-a-Judge.

## Dữ liệu

Dữ liệu gốc: [News Dataset Vietnamese](https://www.kaggle.com/datasets/phamtheds/news-dataset-vietnameses), chủ yếu gồm bài báo tiếng Việt từ Báo Lao Động.

| Manifest | Số mẫu | Mục đích |
|---|---:|---|
| `data/validation_select.jsonl` | 2,000 | Chọn cấu hình decoding |
| `data/test_smoke_10.jsonl` | 10 | Kiểm tra pipeline nhanh |
| `data/test_core_2000.jsonl` | 2,000 | Đánh giá cuối cùng |
| `data/llm_judge_sample_200.jsonl` | 200 | Đánh giá sâu bằng LLM |

Schema mỗi dòng manifest:

```json
{
  "id": "test_000127",
  "split": "test",
  "source": "Nội dung bài báo...",
  "reference": "Tóm tắt tham chiếu..."
}
```

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

## Các hệ thống

### Lead baselines

- **Lead-1:** lấy câu đầu tiên của source.
- **Lead-3:** lấy ba câu đầu tiên của source.
- Cả hai sử dụng cùng bộ tách câu trong `src/baselines.py`.

### Scratch Transformer

- Transformer encoder-decoder Pre-LN huấn luyện từ đầu.
- 6 encoder layers, 6 decoder layers, `d_model=384`, 6 attention heads, `d_ff=1536`.
- SentencePiece unigram, vocabulary 16,000.
- Shared encoder/decoder embedding và tied output projection.
- Cấu hình decoding cuối: beam size 4, length penalty 1.1, no-repeat n-gram size 3.

### ViT5

- Checkpoint: [`VietAI/vit5-base-vietnews-summarization`](https://huggingface.co/VietAI/vit5-base-vietnews-summarization).
- Không thêm prefix ở inference.
- Cấu hình cuối: beam size 4, length penalty 1.1, `max_new_tokens=128`, no-repeat n-gram size 3.

## Cấu trúc repository

```text
vietnamese-summarization/
├── configs/                         # Schema và cấu hình đánh giá
├── data/                            # Manifest đã chuẩn hóa
├── docs/                            # Context, quyết định, run log, hướng dẫn LLM judge
├── notebook/                        # Notebook huấn luyện, inference và tính CR
├── outputs/
│   ├── data_checks/                 # Báo cáo kiểm tra dữ liệu và checksum
│   ├── metrics/                     # Kết quả validation/test
│   ├── predictions/                 # Prediction của bốn hệ thống
│   └── llm_judge/                   # Input, output và phân tích LLM judge
├── report/                          # Slide, hình và báo cáo chất lượng reference
├── scratch_transformer_inference/   # Gói inference Transformer có thể chạy độc lập
├── src/                             # Baseline, evaluator và pipeline đánh giá
├── tests/                           # Kiểm thử pipeline LLM judge
└── README.md
```

## Cài đặt

```bash
git clone https://github.com/ok124-droid/vietnamese-summarization.git
cd vietnamese-summarization

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r scratch_transformer_inference/requirements.txt
pip install pandas matplotlib underthesea transformers accelerate
```

Checkpoint Scratch Transformer có kích thước lớn và không được lưu trực tiếp trong Git. Xem hướng dẫn tải và checksum tại `scratch_transformer_inference/checkpoint_info.txt`.

## Chạy nhanh

### 1. Kiểm tra manifest

```bash
python src/validate_data.py
python src/verify_sha256.py
```

### 2. Sinh Lead-1 và Lead-3

```bash
python src/baselines.py \
  --input data/test_core_2000.jsonl \
  --lead1_out outputs/predictions/test/lead1.jsonl \
  --lead3_out outputs/predictions/test/lead3.jsonl
```

### 3. Chạy Scratch Transformer

```bash
python scratch_transformer_inference/src/infer.py \
  --input data/test_core_2000.jsonl \
  --output outputs/predictions/test/scratch_transformer_test_core_2000.jsonl \
  --checkpoint scratch_transformer_inference/checkpoints/transformer_base/best_val_loss.pt \
  --tokenizer scratch_transformer_inference/data/tokenizer/vietnamese_spm.model \
  --config scratch_transformer_inference/configs/transformer_summarization.yaml \
  --decoding-config scratch_transformer_inference/configs/decoding_beam4_lp1.1_nr3.json \
  --batch-size 4 --device cuda --precision fp32 --overwrite
```

Sau inference, kiểm tra schema và độ bao phủ ID:

```bash
python scratch_transformer_inference/src/validate_predictions.py \
  --manifest data/test_core_2000.jsonl \
  --predictions outputs/predictions/test/scratch_transformer_test_core_2000.jsonl \
  --expected-config-id beam4_lp1.1_nr3
```

### 4. Tính ROUGE và compression ratio

```bash
python src/evaluate.py \
  --manifest data/test_core_2000.jsonl \
  --predictions_dir outputs/predictions/test \
  --output_json outputs/metrics/test_eval.json
```

## Giao thức LLM-as-a-Judge

- Đánh giá paired trên cùng 200 source và bốn candidate.
- Không cung cấp reference khi chấm prediction.
- Ẩn tên hệ thống và cân bằng vị trí A/B/C/D.
- Dùng cùng prompt/rubric cho toàn bộ batch.
- Chấm năm tiêu chí từ 1 đến 5: Faithfulness, Coverage, Focus/Relevance, Coherence/Fluency và Conciseness/Non-redundancy.
- Tính khoảng tin cậy bootstrap 95% trên `sample_id` và paired comparison giữa các hệ thống.

Chi tiết xem `docs/LLM_AS_JUDGE_GUIDE.md` và notebook trong `outputs/llm_judge/đánh giá chất lượng prediction/`.

## Cách diễn giải kết quả

Không nên chọn một hệ thống chỉ từ một chỉ số:

- **Scratch Transformer** tốt nhất theo ROUGE.
- **Lead-3** tốt nhất theo overall của LLM judge, nhưng đầu ra dài hơn rõ rệt.
- **ViT5** súc tích và trôi chảy hơn Scratch Transformer, nhưng factuality chưa tốt trên mẫu đánh giá.
- **Scratch Transformer và ViT5** chưa khác biệt rõ ràng về overall khi xét paired bootstrap CI.
- Reference có lỗi hỗ trợ thông tin, vì vậy ROUGE không phải ground truth tuyệt đối.

## Tài liệu

- `docs/PROJECT_CONTEXT.md`: bối cảnh và quy ước ổn định.
- `docs/DECISIONS.md`: các quyết định thực nghiệm.
- `docs/RUN_LOG.md`: lịch sử các lần chạy.
- `docs/LLM_AS_JUDGE_GUIDE.md`: rubric và giao thức đánh giá.
- `report/REFERENCE_SOURCE_QUALITY_REPORT.md`: phân tích chất lượng source/reference.
- `report/Slide_NLP.pdf`: slide tổng kết dự án.

## Hạn chế và hướng phát triển

- Candidate evaluation hiện dựa trên một LLM judge và 200 mẫu; nên audit một tập con khó bằng người hoặc judge thứ hai khi có điều kiện.
- Lead baselines hưởng lợi từ lead bias của văn phong báo chí và gần như không tạo thông tin mới.
- Hai mô hình sinh vẫn có tỷ lệ factual error cao; hướng tiếp theo là factuality-aware fine-tuning, contrastive negative samples, reranking hoặc post-generation verification.
- Cần chuẩn hóa lại `requirements.txt` ở thư mục gốc và hợp nhất các đường dẫn/config cũ trước khi phát hành một bản release tái lập hoàn toàn.

## Lưu ý an toàn

Không commit API key, checkpoint lớn hoặc file ánh xạ ẩn danh dùng trong quá trình blind evaluation. Các file này nên được lưu ngoài repository công khai và chỉ chia sẻ bằng kênh riêng khi cần audit.

## License

Repository hiện chưa công bố giấy phép sử dụng. Hãy bổ sung `LICENSE` trước khi tái sử dụng hoặc phân phối dự án ra bên ngoài phạm vi học tập.
