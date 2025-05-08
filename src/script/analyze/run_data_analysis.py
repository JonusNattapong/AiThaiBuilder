import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# เพิ่ม src directory ลงใน path เพื่อให้ import modules จาก src ได้
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, src_dir)

from script.analyze.data_explainer import DataExplainer

def main():
    parser = argparse.ArgumentParser(
        description="RunThaiGenDataset Data Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input_file", type=str, required=True, 
                      help="ไฟล์ CSV, JSON, หรือ JSONL ที่ต้องการวิเคราะห์")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                      help="ไดเรกทอรีสำหรับบันทึกผลการวิเคราะห์")
    parser.add_argument("--text_column", type=str, default="text",
                      help="ชื่อคอลัมน์ที่มีข้อความที่ต้องการวิเคราะห์")
    parser.add_argument("--label_column", type=str, default=None,
                      help="ชื่อคอลัมน์ที่มี label (ถ้ามี)")
    parser.add_argument("--format", type=str, choices=["html", "json"], default="html",
                      help="รูปแบบรายงานผลการวิเคราะห์")
    parser.add_argument("--visualize_only", action="store_true", 
                      help="สร้างเฉพาะการแสดงผลแบบภาพ ไม่วิเคราะห์เชิงลึก")
    parser.add_argument("--analyze_similarity", action="store_true",
                      help="วิเคราะห์ความคล้ายคลึงระหว่างข้อความ (อาจใช้เวลานาน)")
    
    args = parser.parse_args()
    
    # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
    if not os.path.exists(args.input_file):
        print(f"Error: ไม่พบไฟล์ '{args.input_file}'")
        return
    
    # ทำให้เส้นทางเป็น Path object เพื่อความสะดวกในการจัดการ
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # สร้างชื่อไดเรกทอรีผลลัพธ์ที่เหมาะสม
    if output_dir.name == "analysis_results":
        output_dir = output_dir / input_file.stem
    
    print(f"\n=== การวิเคราะห์ข้อมูล: {input_file.name} ===")
    print(f"ไดเรกทอรีผลลัพธ์: {output_dir}")
    print(f"คอลัมน์ข้อความ: {args.text_column}")
    if args.label_column:
        print(f"คอลัมน์ label: {args.label_column}")
    print("=" * 50)
    
    # สร้าง DataExplainer
    explainer = DataExplainer(
        data=str(input_file),
        text_column=args.text_column,
        label_column=args.label_column,
        output_dir=str(output_dir)
    )
    
    # สร้างกราฟพื้นฐาน
    print("\nกำลังวิเคราะห์ข้อมูลเบื้องต้น...")
    
    stats = explainer.basic_statistics()
    print(f"จำนวนตัวอย่าง: {stats['total_samples']}")
    print(f"ความยาวข้อความเฉลี่ย: {stats['text_length']['mean']:.2f} ตัวอักษร")
    print(f"จำนวนคำเฉลี่ย: {stats['word_count']['mean']:.2f} คำ")
    
    # สร้างกราฟพื้นฐาน
    print("\nกำลังสร้างกราฟพื้นฐาน...")
    explainer.plot_length_distribution()
    
    if args.label_column:
        print("กำลังวิเคราะห์การกระจายของ label...")
        explainer.plot_label_distribution()
    
    print("กำลังสร้าง word cloud...")
    explainer.create_wordcloud()
    
    print("กำลังวิเคราะห์คำที่พบบ่อย...")
    vocab_data = explainer.analyze_vocabulary()
    explainer.plot_word_frequency()
    print(f"ขนาดคลังคำศัพท์: {vocab_data['vocabulary_size']} คำ")
    
    # ถ้าไม่ใช่แค่สร้างภาพ ให้วิเคราะห์เชิงลึก
    if not args.visualize_only:
        print("\nกำลังวิเคราะห์ข้อมูลเชิงลึก...")
        
        print("กำลังวิเคราะห์การกระจายของชนิดคำ (POS)...")
        explainer.analyze_pos_distribution()
        
        print("กำลังวิเคราะห์ความซับซ้อนของเนื้อหา...")
        explainer.analyze_content_complexity()
        
        if args.label_column:
            print("กำลังวิเคราะห์ความสัมพันธ์ระหว่าง label และข้อความ...")
            explainer.analyze_label_correlation()
        
        if args.analyze_similarity and len(explainer.data) > 1:
            print("กำลังวิเคราะห์ความคล้ายคลึงระหว่างข้อความ (อาจใช้เวลานาน)...")
            explainer.analyze_text_similarity()
    
    # สร้างรายงาน
    print("\nกำลังสร้างรายงานสรุป...")
    report_path = explainer.generate_report(output_format=args.format)
    
    print(f"\nการวิเคราะห์เสร็จสมบูรณ์!")
    print(f"รายงานถูกบันทึกไว้ที่: {report_path}")
    print(f"กราฟและการวิเคราะห์ทั้งหมดอยู่ในโฟลเดอร์: {output_dir}")

if __name__ == "__main__":
    main()
