# Báo cáo ngắn: Đánh giá chất lượng `source` và `reference` của tập test

## 1. Phạm vi và dữ liệu sử dụng

Báo cáo đánh giá tập `test_core_2000.jsonl`, tập trung vào mẫu 200 bản ghi đã được chọn để LLM-as-a-Judge kiểm tra chất lượng reference.

- Mẫu đánh giá: 200 bản ghi, lấy đều 50 mẫu từ bốn nhóm độ dài `source`.
- Seed lấy mẫu: `2026`.
- Mô hình giám khảo: `gemini/gemma-4-31b-it`.
- Phiên bản prompt: `v2`.
- Kết quả chạy chính thức: 200/200 bản ghi thành công, không có bản ghi thất bại.
- Commit được phân tích: `75a8cc31c0d9ac02d6c0091647faa2d933851d11`, nhánh `llm-judge-samples`.

Nguồn định lượng chính là `outputs/llm_judge/reference_scores.jsonl`. Để kiểm tra cấu trúc của source, báo cáo đối chiếu thêm `data/llm_judge_sample_200.jsonl` và `outputs/data_checks/manifest_validation_report.json` trên cùng nhánh.

## 2. Chất lượng của `source`

### 2.1. Tính hợp lệ và độ sạch

Kết quả kiểm tra toàn bộ `test_core_2000` cho thấy:

- Đủ 2.000/2.000 bản ghi hợp lệ.
- Không có lỗi schema hoặc cảnh báo sau bước chuẩn hóa.
- Không có `source` hoặc `reference` rỗng.
- Độ dài `source`: tối thiểu 101 từ, trung vị 439 từ và tối đa 1.193 từ.
- Độ dài `reference`: tối thiểu 12 từ, trung vị 39 từ và tối đa 130 từ.

Trong riêng mẫu 200 bản ghi dùng cho LLM judge:

- Có 200 ID duy nhất và không có `source` trùng hoàn toàn.
- Không phát hiện thẻ HTML, ký tự điều khiển, ký tự Unicode thay thế `�` hoặc văn bản không ở dạng NFC.
- Không có `source` rỗng; độ dài trung vị là 434 từ.
- Có 9/200 source kết thúc bằng dấu ba chấm. Đây có thể là cách biên tập hoặc dấu hiệu trích đoạn; chỉ từ dấu ba chấm chưa đủ kết luận văn bản bị cắt sai.

Nhìn chung, `source` có chất lượng kỹ thuật tốt và đủ phù hợp làm đầu vào cho các mô hình tóm tắt. Tuy nhiên, một số source vẫn giữ tiêu đề phụ, tên tác giả hoặc dấu ba chấm của bài báo. Các hiện tượng này không làm hỏng dữ liệu nhưng cho thấy văn bản chưa hoàn toàn đồng nhất về hình thức biên tập.

### 2.2. Giới hạn của kết luận về source

Các phép kiểm tra trên chỉ xác nhận source sạch và hợp lệ về cấu trúc. Chúng không chứng minh source là toàn bộ bài báo gốc hay chứa đầy đủ tiêu đề, ngày đăng và thông tin mở đầu. Vì vậy, khi reference có chi tiết không xuất hiện trong source, không nên tự động kết luận source hoặc reference chắc chắn sai mà cần xem lại bài báo gốc nếu có thể.

## 3. Chất lượng của `reference`

LLM judge đối chiếu từng khẳng định trong reference với source và sử dụng bốn nhãn:

- `fully_supported`: mọi ý cốt lõi được source hỗ trợ.
- `partially_supported`: có cả ý được hỗ trợ và ý không được hỗ trợ.
- `unsupported`: phần lớn khẳng định không có căn cứ trong source.
- `mismatched`: reference nói về sự kiện hoặc bài báo khác.

### 3.1. Kết quả tổng hợp

| Nhãn | Số mẫu | Tỷ lệ | Khoảng tin cậy Wilson 95% |
|---|---:|---:|---:|
| `fully_supported` | 108 | 54,0% | 47,1%–60,8% |
| `partially_supported` | 75 | 37,5% | 31,1%–44,4% |
| `unsupported` | 17 | 8,5% | 5,4%–13,2% |
| `mismatched` | 0 | 0,0% | 0,0%–1,9% |

Như vậy, 92/200 reference, tương đương **46,0%**, chứa ít nhất một ý cốt lõi không được source hỗ trợ. Khoảng tin cậy 95% của tỷ lệ này là 39,2%–52,9%. Trong đó, 17 mẫu thuộc nhóm nghiêm trọng hơn vì phần lớn nội dung reference không có căn cứ trong source.

Kết quả không có mẫu `mismatched` cho thấy reference nhìn chung vẫn nói về đúng chủ đề hoặc sự kiện. Vấn đề chủ yếu là reference bổ sung chi tiết ngoài source hoặc diễn đạt sai một phần, thay vì bị ghép với một bài báo hoàn toàn khác.

### 3.2. Các dạng vấn đề thường gặp

Các lý do do LLM judge trả về cho thấy ba nhóm lỗi nổi bật:

1. **Thêm thời gian, địa điểm hoặc bối cảnh không có trong source.** Ví dụ `test_000535` có nội dung chính đúng nhưng thêm ngày `11.10`; `test_000174` thêm ngày, hội nghị và địa điểm Siem Reap.
2. **Thêm con số, chức danh, tên riêng hoặc nguồn phát biểu.** Nhiều reference nêu giá trị tiền, số người, tên lãnh đạo hoặc tên cơ quan báo chí mà source không cung cấp.
3. **Làm thay đổi ý nghĩa của source.** Ví dụ có mẫu nhầm danh sách ứng cử viên Quốc hội với ứng cử viên HĐND (`test_002212`), hoặc mô tả xu hướng giá trái với source (`test_005107`).

Kết quả theo bốn nhóm độ dài source cũng đáng chú ý:

| Nhóm độ dài source | `fully_supported` | `partially_supported` | `unsupported` |
|---|---:|---:|---:|
| Ngắn nhất | 48% | 38% | 14% |
| Trung bình ngắn | 56% | 34% | 10% |
| Trung bình dài | 52% | 42% | 6% |
| Dài nhất | 60% | 36% | 4% |

Tỷ lệ `unsupported` giảm từ 14% ở nhóm source ngắn nhất xuống 4% ở nhóm dài nhất. Một cách giải thích hợp lý là source ngắn dễ thiếu phần tiêu đề, ngày tháng hoặc bối cảnh mà reference sử dụng. Đây mới là suy luận từ kết quả, chưa phải bằng chứng xác định nguyên nhân; cần kiểm tra bài báo gốc hoặc đánh giá con người để xác nhận.

## 4. Hạn chế của phép đánh giá

- Kết quả reference dựa trên một mô hình giám khảo duy nhất, chưa phải nhãn chuẩn của con người.
- Giám khảo trả `confidence=5` cho cả 200 mẫu, nên trường confidence không giúp phân biệt trường hợp dễ và khó, đồng thời cho thấy độ tự tin của mô hình chưa được hiệu chỉnh tốt.
- LLM judge chỉ kiểm tra reference có được source hỗ trợ hay không; nó không xác minh tính đúng đắn của source với thế giới thực.
- Mẫu 200 được phân tầng theo độ dài và bao phủ tốt các nhóm source, nhưng vẫn chỉ là 10% của `test_core_2000`.

## 5. Kết luận và khuyến nghị

`Source` của tập test có chất lượng kỹ thuật tốt: dữ liệu hợp lệ, sạch, không rỗng và đủ dài để làm đầu vào cho mô hình. Rủi ro còn lại là một số bài có thể là trích đoạn hoặc thiếu metadata của bài báo gốc.

`Reference` có chất lượng không đồng đều. Chỉ 54% reference được source hỗ trợ đầy đủ, còn 46% có ít nhất một chi tiết không có căn cứ trong source. Vì vậy, ROUGE vẫn có thể dùng để so sánh tự động trên cùng tập dữ liệu, nhưng không nên được xem là thước đo duy nhất về chất lượng tóm tắt.

Khuyến nghị cho dự án:

1. Báo cáo ROUGE trên toàn bộ tập test để giữ khả năng so sánh giữa các hệ thống.
2. Đánh giá LLM và con người trực tiếp dựa trên source, không coi reference là đáp án tuyệt đối.
3. Kiểm tra thủ công các mẫu `unsupported` và một phần mẫu `partially_supported`, ưu tiên lỗi ngày tháng, địa điểm, con số và thực thể.
4. Nếu còn bài báo gốc, kiểm tra liệu tiêu đề hoặc metadata đã bị loại khỏi source hay reference thực sự chứa thông tin sai.
5. Khi trình bày kết quả, nêu rõ hạn chế của reference để tránh kết luận rằng mô hình kém chỉ vì không trùng với một reference có lỗi.

## Tài liệu và tệp nguồn

- Repository: <https://github.com/ok124-droid/vietnamese-summarization/tree/llm-judge-samples>
- Kết quả LLM judge: `outputs/llm_judge/reference_scores.jsonl`
- Thông tin lượt chạy: `outputs/llm_judge/reference_run.json`
- Manifest lấy mẫu: `outputs/llm_judge/sampling_manifest.json`
- Mẫu đầu vào: `data/llm_judge_sample_200.jsonl`
- Báo cáo kiểm tra dữ liệu: `outputs/data_checks/manifest_validation_report.json`
