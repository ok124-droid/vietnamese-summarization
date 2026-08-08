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