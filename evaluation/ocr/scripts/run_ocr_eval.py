import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
from src.ingestion.pipeline import IngestionPipeline
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import re

def normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi so sánh: Loại bỏ tất cả noise từ Markdown formatting."""
    if not text:
        return ""
    
    # 1. Chuyển về chữ thường
    text = text.lower()
    
    # 2. Xóa các tag HTML (<br>, <br/>) thường có trong GT
    text = re.sub(r'<br\s*/?>', ' ', text)
    
    # 3. Xóa toàn bộ dòng kẻ bảng Markdown dạng |---|---| hoặc |:---|:---|
    text = re.sub(r'^\|[\s:|-]+\|\s*$', '', text, flags=re.MULTILINE)
    
    # 4. Xóa dấu heading Markdown (#, ##, ###)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # 5. Xóa các ký tự định dạng: |, *, dấu gạch ngang kẻ ----
    text = text.replace("|", " ").replace("*", " ")
    text = re.sub(r'-{3,}', ' ', text)  # --- hoặc --------
    
    # 6. Chỉ xóa dấu gạch ngang '-' nếu nó đứng độc lập, GIỮ nếu nằm trong số âm
    text = re.sub(r'(?<!\d)-(?!\d)', ' ', text)
    
    # 7. Xóa dấu ngoặc vuông dùng cho checkbox [ ] [x]
    text = re.sub(r'\[\s*[xX]?\s*\]', ' ', text)
    
    # 8. Thu gọn khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def calculate_cer(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)

def calculate_wer(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    distance = Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words)

def main():
    parser = argparse.ArgumentParser(description="Colab OCR Evaluation Suite")
    parser.add_argument("--workers", type=int, default=1, help="Số lượng luồng chạy song song (batch processing). Khuyến nghị: 4-8 trên Colab.")
    args = parser.parse_args()

    data_dir = Path("evaluation/ocr/data")
    report_file = Path("evaluation/ocr/eval_report.json")
    
    # Auto-resume logic: Đọc kết quả cũ nếu có
    results = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔄 Auto-Resume: Tìm thấy {len(results)} kết quả cũ. Sẽ BỎ QUA các file này.")
        except json.JSONDecodeError:
            print("⚠️ File eval_report.json bị lỗi, sẽ ghi đè từ đầu.")
            
    pipeline = IngestionPipeline()
    
    print("="*80)
    print("🚀 FIN-SIGHT AI: COLAB OCR EVALUATION SUITE")
    print("="*80)
    
    # Tìm tất cả ảnh trong các thư mục con (cord_v2, sroie, custom_finsight)
    all_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        all_images.extend(list(data_dir.rglob(ext)))
        
    all_images = sorted(all_images)
    
    if not all_images:
        print("❌ Không tìm thấy ảnh nào trong evaluation/ocr/data/")
        return
        
    print(f"Tổng số ảnh trong thư mục: {len(all_images)}")
    
    # Bọc việc ghi file bằng Lock để tránh race condition khi chạy multi-thread
    write_lock = threading.Lock()
    
    def process_image(img_path: Path):
        dataset_name = img_path.parent.name
        img_id = f"{dataset_name}/{img_path.name}"
        
        if img_id in results:
            return None # Bỏ qua nếu đã chấm điểm
            
        # Tìm file Ground Truth với 2 trường hợp phổ biến:
        # ƯU TIÊN 1: Giữ nguyên đuôi ảnh (vd: ảnh.jpg -> ảnh.jpg.gt.txt) — chính xác nhất
        # ƯU TIÊN 2: Bỏ đuôi ảnh (vd: ảnh.jpg -> ảnh.gt.txt) — dễ bị xung đột nếu có 2 ảnh cùng tên
        gt_path_exact = img_path.with_name(img_path.name + '.gt.txt')
        gt_path_stem = img_path.with_suffix('.gt.txt')
        
        gt_path = gt_path_exact if gt_path_exact.exists() else gt_path_stem
        if not gt_path.exists():
            return f"⚠️  Bỏ qua {img_id} (Không có Ground Truth .gt.txt)"
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            
        t0 = time.time()
        try:
            extraction = pipeline.run(img_path)
            latency = time.time() - t0
            
            prediction = (extraction.markdown or "").strip()
            
            # Ghi output ra file để sau này dễ dàng debug/kiểm tra nếu điểm thấp
            pred_path = img_path.with_name(img_path.name + ".pred.md")
            with open(pred_path, 'w', encoding='utf-8') as f:
                f.write(prediction)

            cer = calculate_cer(ground_truth, prediction)
            wer = calculate_wer(ground_truth, prediction)
            
            # ĐÃ SỬA: Dùng normalized text cho Sim (trước đây dùng raw text gây trừ điểm oan)
            norm_gt = normalize_text(ground_truth)
            norm_pred = normalize_text(prediction)
            sim = fuzz.ratio(norm_gt, norm_pred) / 100.0
            
            # Token Sort Ratio: Bỏ qua thứ tự từ (quan trọng cho bảng biểu)
            sim_token = fuzz.token_sort_ratio(norm_gt, norm_pred) / 100.0
            
            with write_lock:
                results[img_id] = {
                    "dataset": dataset_name,
                    "cer": round(cer, 3),
                    "wer": round(wer, 3),
                    "sim": round(sim, 3),
                    "sim_token": round(sim_token, 3),
                    "latency": round(latency, 2),
                    "mode": extraction.metadata.get('layout_mode')
                }
                # Ghi file liên tục (Checkpointing)
                with open(report_file, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            
            return f"✅ Xong {img_id} ({latency:.2f}s) | CER: {cer:.3f} | WER: {wer:.3f} | Sim: {sim:.2%} | TokenSim: {sim_token:.2%}"
        except Exception as e:
            return f"❌ LỖI tại {img_id}: {e}"

    processed_this_session = 0
    if args.workers > 1:
        print(f"⚡ Bật chế độ chạy song song với {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_image, img_path) for img_path in all_images]
            for future in as_completed(futures):
                msg = future.result()
                if msg:
                    print(msg)
                    processed_this_session += 1
    else:
        print("Chạy tuần tự (1 worker)...")
        for img_path in all_images:
            msg = process_image(img_path)
            if msg:
                print(msg)
                processed_this_session += 1
            
    print("\n" + "="*80)
    print("📈 TỔNG KẾT ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*80)
    
    if results:
        total_cer = sum(r["cer"] for r in results.values())
        total_wer = sum(r["wer"] for r in results.values())
        total_sim = sum(r["sim"] for r in results.values())
        n = len(results)
        
        total_sim_token = sum(r.get("sim_token", r["sim"]) for r in results.values())
        
        # --- GHI RA FILE TXT ĐỂ DỄ DÀNG COPY/PASTE VÀ ĐÁNH GIÁ ---
        txt_report_file = Path("evaluation/ocr/eval_report.txt")
        with open(txt_report_file, "w", encoding="utf-8") as f:
            f.write("BÁO CÁO ĐÁNH GIÁ CHI TIẾT TỪNG ẢNH\n")
            f.write("="*80 + "\n")
            for img_id, r in results.items():
                st = r.get('sim_token', r['sim'])
                f.write(f"[{img_id}] | CER: {r['cer']:.3f} | WER: {r['wer']:.3f} | Sim: {r['sim']:.2%} | TokenSim: {st:.2%}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TỔNG KẾT ĐÁNH GIÁ\n")
            f.write("="*80 + "\n")
            f.write(f"Tổng số file đã hoàn thành : {n}\n")
            f.write(f"CER trung bình toàn tập    : {total_cer/n:.3f}\n")
            f.write(f"WER trung bình toàn tập    : {total_wer/n:.3f}\n")
            f.write(f"Sim trung bình (normalized): {total_sim/n:.2%}\n")
            f.write(f"TokenSim trung bình        : {total_sim_token/n:.2%}\n")

        print(f"Tổng số file đã hoàn thành : {n}")
        print(f"CER trung bình toàn tập    : {total_cer/n:.3f}")
        print(f"WER trung bình toàn tập    : {total_wer/n:.3f}")
        print(f"Sim trung bình (normalized): {total_sim/n:.2%}")
        print(f"TokenSim trung bình        : {total_sim_token/n:.2%}")
        print(f"\n📁 Đã lưu file báo cáo chi tiết tại: {txt_report_file}")
    else:
        print("Chưa có kết quả nào được ghi nhận.")

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    main()
