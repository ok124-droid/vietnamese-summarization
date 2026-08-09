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
