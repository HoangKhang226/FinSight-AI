"""
Script chấm điểm nhẹ — KHÔNG cần PaddleOCR hay VLM.
Chỉ cần file .pred.md (output AI) + .gt.txt (đáp án) là chạy được.

Bao gồm:
  - CER / WER (text-level)
  - Sim / TokenSim (normalized fuzzy matching)  
  - Content Extraction Score (chỉ so sánh số liệu + nhãn tài chính)

Cách dùng:
  python evaluation/ocr/scripts/score_only.py
"""
import sys, re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz


# ============================================================================
# PHẦN 1: Normalize Text (cho CER/WER/Sim)
# ============================================================================

def normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi so sánh: Loại bỏ noise từ Markdown formatting."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'^\|[\s:|-]+\|\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = text.replace("|", " ").replace("*", " ")
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'(?<!\d)-(?!\d)', ' ', text)
    text = re.sub(r'\[\s*[xX]?\s*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    if not ref: return 1.0 if hyp else 0.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def calculate_wer(ref: str, hyp: str) -> float:
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    rw, hw = ref.split(), hyp.split()
    if not rw: return 1.0 if hw else 0.0
    return Levenshtein.distance(rw, hw) / len(rw)


# ============================================================================
# PHẦN 2: Content Extraction Score (Option C)
# ============================================================================

def extract_numbers(text: str) -> list[str]:
    """
    Trích xuất tất cả số liệu tài chính từ text.
    Hỗ trợ các format:
      - 1,000,000 hoặc 1.000.000 (dấu phân cách hàng nghìn)
      - 80,000,000.00 (VN/US style)
      - (461) = số âm trong kế toán
      - -250, $550, 7.5%
    """
    # Chuẩn hóa: bỏ dấu $ và ký tự tiền tệ
    text = text.replace("$", "").replace("₫", "").replace("VND", "").replace("VNĐ", "")
    
    # Xử lý số âm dạng kế toán (461) -> -461
    text = re.sub(r'\((\d[\d,.\s]*)\)', r'-\1', text)
    
    # Tìm tất cả pattern số: hỗ trợ dấu phân cách , hoặc . và phần thập phân
    raw_numbers = re.findall(r'-?(?:\d{1,3}(?:[,.\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', text)
    
    # Chuẩn hóa: loại bỏ dấu phân cách hàng nghìn, chỉ giữ giá trị thuần
    normalized = []
    for num_str in raw_numbers:
        # Bỏ khoảng trắng
        num_str = num_str.replace(" ", "")
        
        # Xác định xem dấu . là thập phân hay phân cách hàng nghìn
        # Nếu có cả , và . -> , là hàng nghìn, . là thập phân (EN style: 1,000.50)
        # Nếu chỉ có . với pattern X.XXX -> . là hàng nghìn (VN style: 1.000)
        if ',' in num_str and '.' in num_str:
            # EN style: 80,000,000.00
            num_str = num_str.replace(",", "")
        elif ',' in num_str:
            # Chỉ có dấu , -> phân cách hàng nghìn
            num_str = num_str.replace(",", "")
        elif re.match(r'-?\d{1,3}(\.\d{3})+$', num_str):
            # VN style: 1.000.000 (dấu . là hàng nghìn, không có phần thập phân)
            num_str = num_str.replace(".", "")
        # Còn lại: giữ nguyên (vd: 7.5, 0.50)
        
        # Loại bỏ leading zeros (nhưng giữ "0" và "0.xx")
        try:
            if '.' in num_str:
                val = float(num_str)
                # Nếu là số nguyên (vd 1000.0 -> 1000)
                if val == int(val) and abs(val) > 1:
                    normalized.append(str(int(val)))
                else:
                    normalized.append(str(val))
            else:
                normalized.append(str(int(num_str)))
        except ValueError:
            normalized.append(num_str)
    
    return normalized


def extract_labels(text: str) -> set[str]:
    """
    Trích xuất các nhãn/thuật ngữ tài chính quan trọng từ text.
    Chỉ giữ lại các cụm từ có ý nghĩa (≥ 2 từ hoặc từ khóa đặc biệt).
    """
    text = normalize_text(text)
    
    # Danh sách từ khóa tài chính quan trọng (1 từ cũng đếm)
    keywords_single = {
        'total', 'subtotal', 'tax', 'balance', 'debit', 'credit',
        'cash', 'equity', 'assets', 'liabilities', 'revenue', 'income',
        'tổng', 'thuế', 'nợ', 'có', 'dư', 'vốn', 'lãi', 'phí',
    }
    
    # Tách thành các từ, lọc bỏ số và ký tự đặc biệt
    words = re.findall(r'[a-zA-ZÀ-ỹ]{2,}', text)
    
    labels = set()
    
    # Thêm từ khóa đơn
    for w in words:
        if w in keywords_single:
            labels.add(w)
    
    # Tạo bigrams (cụm 2 từ liền nhau) để bắt các thuật ngữ phức hợp
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        labels.add(bigram)
    
    # Tạo trigrams cho các thuật ngữ dài (vd: "tài sản ngắn")
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        labels.add(trigram)
    
    return labels


def content_extraction_score(gt_text: str, pred_text: str) -> dict:
    """
    Tính Content Extraction Score: so sánh số liệu + nhãn giữa GT và Prediction.
    
    Returns:
        dict với:
          - number_precision: % số trong pred có mặt trong GT
          - number_recall: % số trong GT được tìm thấy trong pred  
          - number_f1: F1 score cho số
          - label_recall: % nhãn trong GT được tìm thấy trong pred
          - content_score: điểm tổng hợp (trọng số: 60% số + 40% nhãn)
    """
    gt_numbers = extract_numbers(gt_text)
    pred_numbers = extract_numbers(pred_text)
    
    gt_labels = extract_labels(gt_text)
    pred_labels = extract_labels(pred_text)
    
    # --- Chấm điểm SỐ LIỆU ---
    # Dùng Counter để xử lý trường hợp cùng 1 số xuất hiện nhiều lần
    gt_counter = Counter(gt_numbers)
    pred_counter = Counter(pred_numbers)
    
    # Số trùng khớp = min(count_gt, count_pred) cho mỗi số
    matched_numbers = sum((gt_counter & pred_counter).values())
    
    total_gt_nums = len(gt_numbers)
    total_pred_nums = len(pred_numbers)
    
    if total_gt_nums == 0:
        num_recall = 1.0
        num_precision = 1.0
    else:
        num_recall = matched_numbers / total_gt_nums if total_gt_nums > 0 else 0
        num_precision = matched_numbers / total_pred_nums if total_pred_nums > 0 else 0
    
    num_f1 = (2 * num_precision * num_recall / (num_precision + num_recall)) if (num_precision + num_recall) > 0 else 0
    
    # --- Chấm điểm NHÃN ---
    if gt_labels:
        matched_labels = gt_labels & pred_labels
        label_recall = len(matched_labels) / len(gt_labels)
    else:
        label_recall = 1.0
    
    # --- Điểm tổng hợp: Trọng số 60% số liệu + 40% nhãn ---
    content_score = 0.6 * num_f1 + 0.4 * label_recall
    
    return {
        "num_recall": round(num_recall, 3),
        "num_precision": round(num_precision, 3),
        "num_f1": round(num_f1, 3),
        "label_recall": round(label_recall, 3),
        "content_score": round(content_score, 3),
        "gt_nums_count": total_gt_nums,
        "pred_nums_count": total_pred_nums,
        "matched_nums": matched_numbers,
    }


# ============================================================================
# PHẦN 3: Main
# ============================================================================

def main():
    data_dir = Path("evaluation/ocr/data/custom_finsight")
    pred_files = sorted(data_dir.glob("*.pred.md"))
    
    if not pred_files:
        print("❌ Không tìm thấy file .pred.md nào!")
        return
    
    results = []
    
    print("=" * 110)
    print("📊 CHẤM ĐIỂM ĐA CHIỀU (Text Similarity + Content Extraction)")
    print("=" * 110)
    print(f"{'File':<45} | {'Sim':>6} | {'TkSim':>6} | {'NumF1':>6} | {'LblRc':>6} | {'Content':>7} |")
    print("-" * 110)
    
    for pred_file in pred_files:
        img_name = pred_file.name.replace(".pred.md", "")
        
        gt_exact = data_dir / (img_name + ".gt.txt")
        gt_stem = data_dir / (Path(img_name).with_suffix('.gt.txt').name)
        gt_path = gt_exact if gt_exact.exists() else gt_stem
        
        if not gt_path.exists():
            continue
        
        gt = gt_path.read_text(encoding='utf-8').strip()
        pred = pred_file.read_text(encoding='utf-8').strip()
        
        norm_gt = normalize_text(gt)
        norm_pred = normalize_text(pred)
        
        cer = calculate_cer(gt, pred)
        wer = calculate_wer(gt, pred)
        sim = fuzz.ratio(norm_gt, norm_pred) / 100.0
        sim_token = fuzz.token_sort_ratio(norm_gt, norm_pred) / 100.0
        
        # Content Extraction Score
        ce = content_extraction_score(gt, pred)
        
        results.append({
            "name": img_name,
            "cer": cer, "wer": wer,
            "sim": sim, "sim_token": sim_token,
            **ce
        })
        
        # Emoji
        icon = "🟢" if ce["content_score"] >= 0.85 else "🟡" if ce["content_score"] >= 0.70 else "🔴"
        short_name = img_name[:42] + "..." if len(img_name) > 45 else img_name
        print(f"{icon} {short_name:<44} | {sim:>5.1%} | {sim_token:>5.1%} | {ce['num_f1']:>5.1%} | {ce['label_recall']:>5.1%} | {ce['content_score']:>6.1%} |")
    
    if not results:
        print("Không có kết quả nào.")
        return
    
    n = len(results)
    avg = lambda key: sum(r[key] for r in results) / n
    
    print("\n" + "=" * 110)
    print("📈 TỔNG KẾT")
    print("=" * 110)
    print(f"Tổng số file                : {n}")
    print(f"CER trung bình              : {avg('cer'):.3f}")
    print(f"WER trung bình              : {avg('wer'):.3f}")
    print(f"Sim trung bình (normalized) : {avg('sim'):.2%}")
    print(f"TokenSim trung bình         : {avg('sim_token'):.2%}")
    print(f"─── Content Extraction ───")
    print(f"Number F1 trung bình        : {avg('num_f1'):.2%}")
    print(f"Label Recall trung bình     : {avg('label_recall'):.2%}")
    print(f"Content Score trung bình    : {avg('content_score'):.2%}")
    
    # Xuất file txt
    report = Path("evaluation/ocr/eval_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO ĐÁNH GIÁ ĐA CHIỀU — FinSight AI OCR\n")
        f.write("=" * 110 + "\n\n")
        
        for r in results:
            f.write(f"[{r['name']}]\n")
            f.write(f"  Text:    CER={r['cer']:.3f}  WER={r['wer']:.3f}  Sim={r['sim']:.2%}  TokenSim={r['sim_token']:.2%}\n")
            f.write(f"  Content: NumF1={r['num_f1']:.2%} (recall={r['num_recall']:.2%}, precision={r['num_precision']:.2%}, matched={r['matched_nums']}/{r['gt_nums_count']})  LabelRecall={r['label_recall']:.2%}\n")
            f.write(f"  >>> Content Score = {r['content_score']:.2%}\n\n")
        
        f.write("=" * 110 + "\n")
        f.write("TỔNG KẾT\n")
        f.write("=" * 110 + "\n")
        f.write(f"Tổng: {n} files\n")
        f.write(f"CER: {avg('cer'):.3f} | WER: {avg('wer'):.3f} | Sim: {avg('sim'):.2%} | TokenSim: {avg('sim_token'):.2%}\n")
        f.write(f"Number F1: {avg('num_f1'):.2%} | Label Recall: {avg('label_recall'):.2%} | Content Score: {avg('content_score'):.2%}\n")
    
    print(f"\n📁 Đã lưu báo cáo tại: {report}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    main()
