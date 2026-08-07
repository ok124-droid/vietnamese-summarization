import json
import os
import re
import sys
import unicodedata
import argparse
from collections import Counter
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

sys.stdout.reconfigure(encoding='utf-8')

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa Unicode & Chuỗi văn bản:
    1. Chuẩn hóa dạng Unicode NFC (unicodedata.normalize('NFC', text)).
    2. Chuyển thành chữ thường (lowercase).
    3. Xóa các ký tự điều khiển, ký tự trắng thừa.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str, use_segmentation: bool = True) -> list[str]:
    """
    Tokenize văn bản tiếng Việt thành danh sách từ (words).
    - Nếu use_segmentation=True và có underthesea: thực hiện word segmentation ghép từ phức (ví dụ 'thực_hiện').
    - Loại bỏ các ký tự dấu câu độc lập.
    """
    norm_text = normalize_text(text)
    if not norm_text:
        return []
        
    if use_segmentation and HAS_UNDERTHESEA:
        segmented = word_tokenize(norm_text, format="text")
        tokens = [t for t in segmented.split() if re.search(r'\w', t)]
        return tokens
    else:
        tokens = re.findall(r'\w+', norm_text)
        return tokens

def get_ngrams(tokens: list[str], n: int) -> Counter:
    """Tạo N-gram Multiset"""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def compute_rouge_n(pred_tokens: list[str], ref_tokens: list[str], n: int) -> tuple[float, float, float]:
    """
    Tính ROUGE-N (Precision, Recall, F1)
    """
    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    
    total_pred = sum(pred_ngrams.values())
    total_ref = sum(ref_ngrams.values())
    
    if total_pred == 0 or total_ref == 0:
        return 0.0, 0.0, 0.0
        
    overlap_ngrams = pred_ngrams & ref_ngrams
    overlap_count = sum(overlap_ngrams.values())
    
    precision = overlap_count / total_pred
    recall = overlap_count / total_ref
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    return precision, recall, f1

def lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """
    Tính độ dài Longest Common Subsequence (LCS)
    """
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i-1] == seq2[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[n]

def compute_rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> tuple[float, float, float]:
    """
    Tính ROUGE-L (Precision, Recall, F1) dựa trên LCS
    """
    m = len(pred_tokens)
    n = len(ref_tokens)
    
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
        
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    
    precision = lcs_len / m
    recall = lcs_len / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    return precision, recall, f1

def compute_compression_ratio(pred_tokens: list[str], source_tokens: list[str]) -> float:
    """
    Tính Compression Ratio (Tỷ lệ nén):
    Compression Ratio = số_từ(prediction) / số_từ(source)
    """
    len_src = len(source_tokens)
    if len_src == 0:
        return 0.0
    return len(pred_tokens) / len_src

def evaluate_file(manifest_path: str, prediction_path: str, use_segmentation: bool = True) -> dict:
    """
    Ghép nối manifest và prediction theo 'id' và tính các chỉ số trung bình.
    """
    manifest_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                manifest_map[item["id"]] = item
                
    predictions = []
    with open(prediction_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
                
    r1_f1_list, r2_f1_list, rl_f1_list, comp_ratio_list = [], [], [], []
    valid_count = 0
    error_count = 0
    
    for pred_item in predictions:
        item_id = pred_item.get("id")
        status = pred_item.get("status", "ok")
        
        if status != "ok" or item_id not in manifest_map:
            error_count += 1
            continue
            
        manifest_item = manifest_map[item_id]
        source_text = manifest_item.get("source", "")
        reference_text = manifest_item.get("reference", "")
        prediction_text = pred_item.get("prediction", "")
        
        pred_tokens = tokenize_text(prediction_text, use_segmentation=use_segmentation)
        ref_tokens = tokenize_text(reference_text, use_segmentation=use_segmentation)
        source_tokens = tokenize_text(source_text, use_segmentation=use_segmentation)
        
        _, _, r1_f1 = compute_rouge_n(pred_tokens, ref_tokens, n=1)
        _, _, r2_f1 = compute_rouge_n(pred_tokens, ref_tokens, n=2)
        _, _, rl_f1 = compute_rouge_l(pred_tokens, ref_tokens)
        comp_ratio = compute_compression_ratio(pred_tokens, source_tokens)
        
        r1_f1_list.append(r1_f1)
        r2_f1_list.append(r2_f1)
        rl_f1_list.append(rl_f1)
        comp_ratio_list.append(comp_ratio)
        valid_count += 1
        
    avg_r1_f1 = (sum(r1_f1_list) / valid_count * 100) if valid_count > 0 else 0.0
    avg_r2_f1 = (sum(r2_f1_list) / valid_count * 100) if valid_count > 0 else 0.0
    avg_rl_f1 = (sum(rl_f1_list) / valid_count * 100) if valid_count > 0 else 0.0
    avg_comp_ratio = (sum(comp_ratio_list) / valid_count * 100) if valid_count > 0 else 0.0
    
    return {
        "prediction_file": os.path.basename(prediction_path),
        "system": predictions[0].get("system") if predictions else "unknown",
        "total_predictions": len(predictions),
        "valid_count": valid_count,
        "error_count": error_count,
        "rouge_1_f1": round(avg_r1_f1, 4),
        "rouge_2_f1": round(avg_r2_f1, 4),
        "rouge_l_f1": round(avg_rl_f1, 4),
        "compression_ratio_pct": round(avg_comp_ratio, 4)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá ROUGE-1, ROUGE-2, ROUGE-L F1 & Compression Ratio")
    parser.add_argument("--manifest", default="data/test_smoke_10.jsonl", help="Đường dẫn file manifest (source & reference)")
    parser.add_argument("--predictions_dir", default="outputs/predictions/test_smoke_10", help="Thư mục chứa các file prediction")
    parser.add_argument("--output_json", default=None, help="Đường dẫn lưu kết quả JSON (ví dụ: outputs/metrics/test_smoke_10_eval.json)")
    parser.add_argument("--no_segmentation", action="store_true", help="Không sử dụng tách từ tiếng Việt")
    args = parser.parse_args()
    
    use_seg = not args.no_segmentation
    
    print(f"=== ĐÁNH GIÁ CHỈ SỐ BÀI TÓM TẮT VĂN BẢN ===")
    print(f"Manifest: {args.manifest}")
    print(f"Prediction Dir: {args.predictions_dir}")
    print(f"Vietnamese Tokenizer (underthesea): {'Có' if (use_seg and HAS_UNDERTHESEA) else 'Không (Fallback Regex)'}\n")
    
    if os.path.isfile(args.predictions_dir):
        pred_files = [args.predictions_dir]
    elif os.path.isdir(args.predictions_dir):
        pred_files = [os.path.join(args.predictions_dir, f) for f in os.listdir(args.predictions_dir) if f.endswith(".jsonl")]
    else:
        print(f"Lỗi: {args.predictions_dir} không tồn tại!")
        sys.exit(1)
        
    results = []
    for p_file in sorted(pred_files):
        res = evaluate_file(args.manifest, p_file, use_segmentation=use_seg)
        results.append(res)
        
    # In bảng kết quả
    print(f"{'System / File':<28} | {'ROUGE-1 F1':<12} | {'ROUGE-2 F1':<12} | {'ROUGE-L F1':<12} | {'Comp Ratio (%)':<15}")
    print("-" * 88)
    for r in results:
        print(f"{r['prediction_file']:<28} | {r['rouge_1_f1']:<12.4f} | {r['rouge_2_f1']:<12.4f} | {r['rouge_l_f1']:<12.4f} | {r['compression_ratio_pct']:<15.4f}")
        
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu kết quả đánh giá vào: '{args.output_json}'")
