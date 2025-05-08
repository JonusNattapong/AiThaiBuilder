import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# เพิ่ม src directory ลงใน path เพื่อให้ import modules จาก src ได้
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, src_dir)

from script.augment.data_augmenter import ThaiDataAugmenter

def setup_logging(output_dir: str) -> None:
    """ตั้งค่า logging system"""
    log_file = os.path.join(output_dir, 'augmentation.log')
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_input_file(input_file: str) -> bool:
    """ตรวจสอบไฟล์ input"""
    if not os.path.exists(input_file):
        logging.error(f"ไม่พบไฟล์ input: {input_file}")
        return False
        
    ext = os.path.splitext(input_file)[1]
    if ext not in ['.csv', '.tsv', '.json', '.jsonl']:
        logging.error(f"รูปแบบไฟล์ไม่รองรับ: {ext}")
        return False
        
    return True

def get_selected_techniques(args: argparse.Namespace) -> Optional[List[str]]:
    """รวบรวมเทคนิคที่เลือกใช้"""
    if args.all_techniques:
        logging.info("เลือกใช้ทุกเทคนิค")
        return None
        
    techniques = []
    technique_mapping = {
        'random_deletion': args.random_deletion,
        'random_swap': args.random_swap, 
        'random_insertion': args.random_insertion,
        'synonym_replacement': args.synonym_replacement,
        'word_embedding': args.word_embedding,
        'back_translation': args.back_translation,
        'character_replacement': args.character_replacement,
        'contextual_augmentation': args.contextual_augmentation,
        'sentence_augmentation': args.sentence_augmentation
    }
    
    for name, selected in technique_mapping.items():
        if selected:
            techniques.append(name)
            
    if not techniques:
        logging.info("ไม่ได้เลือกเทคนิคใด จะใช้ทุกเทคนิค")
        return None
        
    logging.info(f"เลือกใช้เทคนิค: {', '.join(techniques)}")
    return techniques

def main():
    """ฟังก์ชันหลักสำหรับรัน data augmentation"""
    parser = argparse.ArgumentParser(
        description="RunThaiGenDataset - ตัวช่วยทำ Data Augmentation ภาษาไทย",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument("--input_file", type=str, required=True, 
                      help="ไฟล์ข้อมูลสำหรับ augment (.csv, .tsv, .json, หรือ .jsonl)")
    parser.add_argument("--output_dir", type=str, default="augmented_data",
                      help="ไดเรกทอรีสำหรับบันทึกข้อมูลที่ augment แล้ว")
    parser.add_argument("--output_name", type=str, default=None,
                      help="ชื่อไฟล์ผลลัพธ์ (หากไม่ระบุจะสร้างชื่ออัตโนมัติ)")
    parser.add_argument("--output_format", type=str, choices=["csv", "tsv", "json", "jsonl"],
                      default="csv", help="รูปแบบไฟล์ผลลัพธ์")
    
    # Column settings
    parser.add_argument("--text_column", type=str, default="text",
                      help="ชื่อคอลัมน์ที่มีข้อความที่ต้องการ augment")
    parser.add_argument("--label_column", type=str, default=None,
                      help="ชื่อคอลัมน์ที่มี label (ถ้ามี)")
    
    # Augmentation techniques
    techniques_group = parser.add_argument_group("Augmentation Techniques")
    techniques_group.add_argument("--all_techniques", action="store_true",
                               help="ใช้ทุกเทคนิค (เป็นค่าเริ่มต้น)")
    techniques_group.add_argument("--random_deletion", action="store_true",
                               help="ใช้เทคนิคลบคำสุ่ม")
    techniques_group.add_argument("--random_swap", action="store_true",
                               help="ใช้เทคนิคสลับคำสุ่ม")
    techniques_group.add_argument("--random_insertion", action="store_true",
                               help="ใช้เทคนิคแทรกคำสุ่ม")
    techniques_group.add_argument("--synonym_replacement", action="store_true",
                               help="ใช้เทคนิคแทนที่ด้วยคำพ้องความหมาย")
    techniques_group.add_argument("--word_embedding", action="store_true",
                               help="ใช้เทคนิคแทนที่ด้วยคำที่มี embedding ใกล้เคียง")
    techniques_group.add_argument("--back_translation", action="store_true",
                               help="ใช้เทคนิคแปลภาษาไปกลับ")
    techniques_group.add_argument("--character_replacement", action="store_true",
                               help="ใช้เทคนิคแทนที่ตัวอักษรบางตัว")
    techniques_group.add_argument("--contextual_augmentation", action="store_true",
                               help="ใช้เทคนิค contextual augmentation ด้วย BERT")
    techniques_group.add_argument("--sentence_augmentation", action="store_true",
                               help="ใช้เทคนิค augment ระดับประโยค")
    
    # Augmentation parameters
    aug_params = parser.add_argument_group("Augmentation Parameters")
    aug_params.add_argument("--num_per_technique", type=int, default=1,
                         help="จำนวน augment ที่ต้องการสร้างต่อเทคนิคต่อตัวอย่าง")
    aug_params.add_argument("--balance_labels", action="store_true",
                         help="สร้างข้อมูลเพิ่มให้แต่ละ label มีจำนวนเท่ากัน")
    aug_params.add_argument("--min_length", type=int, default=10,
                         help="ความยาวขั้นต่ำของข้อความที่จะทำ augment")
    aug_params.add_argument("--include_original", action="store_true", default=True,
                         help="รวมข้อความต้นฉบับไว้ในผลลัพธ์")
    aug_params.add_argument("--random_seed", type=int, default=42,
                         help="Random seed สำหรับความสม่ำเสมอ")
    
    args = parser.parse_args()
    
    # ตรวจสอบเทคนิคที่เลือก
    selected_techniques = []
    
    # ตั้งค่า logging
    setup_logging(args.output_dir)
    
    # ตรวจสอบไฟล์ input
    if not validate_input_file(args.input_file):
        sys.exit(1)
        
    # เลือกเทคนิคที่จะใช้
    selected_techniques = get_selected_techniques(args)
    
    # แสดงข้อมูลการทำงาน
    print("\n" + "=" * 50)
    print("THAI DATA AUGMENTATION")
    print("=" * 50)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Text column: {args.text_column}")
    if args.label_column:
        print(f"Label column: {args.label_column}")
    
    print("\nAugmentation Techniques:")
    if selected_techniques is None:
        print("- All available techniques")
    else:
        for technique in selected_techniques:
            print(f"- {technique}")
    
    print(f"\nAugmentations per technique: {args.num_per_technique}")
    print(f"Balance labels: {args.balance_labels}")
    print(f"Minimum text length: {args.min_length}")
    print(f"Include original texts: {args.include_original}")
    print("=" * 50 + "\n")
    
    # สร้าง augmenter และทำการ augment
    try:
        # สร้างและกำหนดค่า augmenter
        augmenter = ThaiDataAugmenter(
            data=args.input_file,
            text_column=args.text_column,
            label_column=args.label_column,
            output_dir=args.output_dir,
            random_state=args.random_seed
        )
        
        # Augment และบันทึกข้อมูล
        augment_params = {
            'num_per_technique': args.num_per_technique,
            'balance_labels': args.balance_labels,
            'min_length': args.min_length,
            'output_filename': args.output_name,
            'include_original': args.include_original,
            'format': args.output_format
        }
        
        if selected_techniques is None:
            output_path = augmenter.augment_and_save(**augment_params)
        else:
            output_path = augmenter.augment_and_save(
                techniques=selected_techniques,
                **augment_params
            )
        
        print(f"\nการทำ Data Augmentation เสร็จสมบูรณ์!")
        print(f"ผลลัพธ์ถูกบันทึกไว้ที่: {output_path}")
        print(f"จำนวนเทคนิคที่ใช้: {len(selected_techniques) if selected_techniques else 'ทั้งหมด'}")
        
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
