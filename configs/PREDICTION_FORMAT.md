# Định dạng chuẩn của file Prediction

Document quy định cấu trúc và thuộc tính chuẩn cho các file prediction thu được từ các mô hình / hệ thống tóm tắt văn bản.

## Dạng chuẩn của file prediction

```json
{
  "id": "test_000127",
  "system": "scratch_transformer",
  "config_id": "beam4_lp1.1",
  "prediction": "Bản tóm tắt do mô hình sinh...",
  "status": "ok",
  "error": null
}
```

## Các trường thông tin

| Trường | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `id` | Có | Khớp chính xác với manifest |
| `system` | Có | Tên hệ thống |
| `config_id` | Có | Cấu hình giải mã |
| `prediction` | Có | Tóm tắt được sinh |
| `status` | Có | `ok` hoặc `error` |
| `error` | Có | `null` nếu thành công, mô tả nếu lỗi |

## Tên hệ thống được phép dùng (`system`)

```
lead1
lead3
scratch_transformer
vit5_large
vit5_base
textrank
```

## Các ví dụ

### Ví dụ Lead-1:

```json
{
  "id": "test_000127",
  "system": "lead1",
  "config_id": "lead1",
  "prediction": "Câu đầu tiên của bài báo.",
  "status": "ok",
  "error": null
}
```

### Ví dụ khi suy luận lỗi:

```json
{
  "id": "test_000391",
  "system": "vit5_large",
  "config_id": "beam4_lp1.0",
  "prediction": "",
  "status": "error",
  "error": "CUDA out of memory"
}
```

## Nguyên tắc lưu trữ & Đánh giá

- Không cần lặp lại `source` và `reference` trong file prediction.
- Evaluator sẽ tự động ghép nối thông tin từ manifest và prediction bằng trường `id`.
