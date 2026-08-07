import json
import os
import re
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8')

def split_sentences(text: str) -> list[str]:
    """
    Tách câu cho văn bản tiếng Việt dựa trên quy tắc bảo vệ từ viết tắt, tên miền, số và ngày tháng.
    
    Quy trình:
    1. Bảo vệ tên miền trang web (ví dụ: laodong.vn, google.com)
    2. Bảo vệ ngày tháng (1.10.2019) và số thập phân / phân cách hàng ngàn (1.234,5)
    3. Bảo vệ các từ chức danh/danh xưng (TS., BS., GS., PGS., ThS., KS., TP.) trước tên riêng
    4. Xử lý dấu ba chấm (... hoặc …): giữ nguyên nếu theo sau là chữ viết thường, ngắt câu nếu theo sau là chữ hoa/chữ số
    5. Đánh dấu và tách câu tại [.!?…] (kèm dấu ngoặc đóng ” " nếu có) theo sau bởi khoảng trắng/xuống dòng và chữ hoa/chữ số.
    """
    if not text or not text.strip():
        return []
        
    text = text.strip()
    
    # 1. Bảo vệ tên miền (ví dụ: laodong.vn, google.com, vnexpress.net)
    protected = re.sub(r'\b([a-zA-Z0-9-]+)\.([a-zA-Z]{2,4})\b', r'\1<DOT>\2', text)
    
    # 2. Bảo vệ ngày tháng và số (ví dụ: 1.10.2019, 1.234,5)
    protected = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', protected)
    
    # 3. Bảo vệ từ viết tắt danh xưng/chức danh trước tên riêng
    titles = r'(?:BS|TS|GS|PGS|ThS|KS|NSND|NSƯT)'
    protected = re.sub(rf'\b({titles})\.\s+(?=[A-ZÀÁẢẠÃÂẦẤẨẬẪĂẰẮẲẶẴÈÉẺẸẼÊỀẾỂỆỄÌÍỈỊĨÒÓỎỌÕÔỒỐỔỘỖƠỜỚỞỢỠÙÚỦỤŨƯỪỨỬỰỮỲÝỶỴỸĐ])', r'\1<DOT> ', protected)
    
    # Bảo vệ TP.HCM, TP. Đà Nẵng
    protected = re.sub(r'\bTP\.HCM\b', r'TP<DOT>HCM', protected, flags=re.IGNORECASE)
    protected = re.sub(r'\bTP\.\s+(?=[A-ZÀÁẢẠÃÂẦẤẨẬẪĂẰẮẲẶẴÈÉẺẸẼÊỀẾỂỆỄÌÍỈỊĨÒÓỎỌÕÔỒỐỔỘỖƠỜỚỞỢỠÙÚỦỤŨƯỪỨỬỰỮỲÝỶỴỸĐ])', r'TP<DOT> ', protected)
    
    # 4. Xử lý dấu ba chấm (... hoặc …)
    def protect_ellipsis_lowercase(m):
        return m.group(1) + '<SPACE>' + m.group(2)
        
    protected = re.sub(r'(\.\.\.|…)\s+([a-zàáảạãâầấẩậẫăằắẳặẵèéẻẹẽêềếểệễìíỉịĩòóỏọõôồốổộỗơờớởợỡùúủụũưừứửựữỳýỷỵỹđ])', protect_ellipsis_lowercase, protected)
    
    # 5. Đánh dấu vị trí kết thúc câu bằng <SENT_END>
    upper_chars = r'A-ZÀÁẢẠÃÂẦẤẨẬẪĂẰẮẲẶẴÈÉẺẸẼÊỀẾỂỆỄÌÍỈỊĨÒÓỎỌÕÔỒỐỔỘỖƠỜỚỞỢỠÙÚỦỤŨƯỪỨỬỰỮỲÝỶỴỸĐ0-9“"\('
    pattern = rf'([.!?…]["”\']?)(\s+|\n+)(?=[{upper_chars}]|$)'
    marked = re.sub(pattern, r'\1<SENT_END>', protected)
    
    # Xuống dòng nếu chưa được đánh dấu
    marked = re.sub(r'\n+', '<SENT_END>', marked)
    
    raw_chunks = marked.split('<SENT_END>')
    
    sentences = []
    for chunk in raw_chunks:
        cleaned = chunk.replace('<DOT>', '.').replace('<SPACE>', ' ').strip()
        if cleaned:
            sentences.append(cleaned)
            
    return sentences

def generate_lead_prediction(item: dict, lead_n: int) -> dict:
    """
    Sinh record prediction cho Lead-N theo chuẩn configs/prediction_schema.json
    """
    item_id = item.get("id")
    source_text = item.get("source", "")
    system_name = f"lead{lead_n}"
    config_id = f"lead{lead_n}"
    
    try:
        sentences = split_sentences(source_text)
        if sentences:
            prediction_text = " ".join(sentences[:lead_n])
        else:
            prediction_text = source_text
            
        return {
            "id": item_id,
            "system": system_name,
            "config_id": config_id,
            "prediction": prediction_text,
            "status": "ok",
            "error": None
        }
    except Exception as e:
        return {
            "id": item_id,
            "system": system_name,
            "config_id": config_id,
            "prediction": "",
            "status": "error",
            "error": str(e)
        }

def process_file(input_path: str, lead_1_output: str, lead_3_output: str):
    """
    Đọc file JSONL đầu vào, sinh Lead-1 và Lead-3, lưu kết quả ra file chuẩn.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    lead1_preds = [generate_lead_prediction(rec, 1) for rec in records]
    lead3_preds = [generate_lead_prediction(rec, 3) for rec in records]
    
    os.makedirs(os.path.dirname(lead_1_output), exist_ok=True)
    os.makedirs(os.path.dirname(lead_3_output), exist_ok=True)
    
    with open(lead_1_output, "w", encoding="utf-8") as f:
        for pred in lead1_preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            
    with open(lead_3_output, "w", encoding="utf-8") as f:
        for pred in lead3_preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            
    print(f"Đã xử lý {len(records)} mẫu từ '{input_path}'")
    print(f"File Lead-1: '{lead_1_output}'")
    print(f"File Lead-3: '{lead_3_output}'")
    return lead1_preds, lead3_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh Lead-1 và Lead-3 baseline summarization")
    parser.add_argument("--input", default="data/test_smoke_10.jsonl", help="Đường dẫn file jsonl đầu vào")
    parser.add_argument("--lead1_out", default="outputs/predictions/lead1_smoke_10.jsonl", help="Output file Lead-1")
    parser.add_argument("--lead3_out", default="outputs/predictions/lead3_smoke_10.jsonl", help="Output file Lead-3")
    args = parser.parse_args()
    
    process_file(args.input, args.lead1_out, args.lead3_out)
