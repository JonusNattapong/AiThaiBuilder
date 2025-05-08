import os
import sys
import argparse
import pandas as pd
from datasets import load_dataset

# เพิ่ม src directory ลงใน path เพื่อให้ import modules จาก src ได้
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, src_dir)

from script.train.model_fine_tuner import ModelFineTuner, SUPPORTED_MODELS, TASK_CONFIGS

def get_available_models():
    """แสดงรายการโมเดลที่รองรับ"""
    print("\nSupported Models:")
    print("=" * 50)
    for model_name, config in SUPPORTED_MODELS.items():
        sizes = [size for size in config.keys() if size not in ["type", "tokenizer_kwargs"]]
        print(f"- {model_name}: {', '.join(sizes)}")

def get_available_tasks():
    """แสดงรายการงานที่รองรับ"""
    print("\nSupported Tasks:")
    print("=" * 50)
    for task_name, config in TASK_CONFIGS.items():
        metrics = config.get("metrics", [])
        print(f"- {task_name}: {', '.join(metrics)}")

def main():
    parser = argparse.ArgumentParser(
        description="RunThaiGenDataset Fine-Tuning Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--list_models", action="store_true", help="แสดงรายการโมเดลที่รองรับ")
    parser.add_argument("--list_tasks", action="store_true", help="แสดงรายการงานที่รองรับ")
    
    # Required arguments
    parser.add_argument("--model", type=str, help="ชื่อโมเดล (เช่น wangchanberta, mt5)")
    parser.add_argument("--model_size", type=str, default="base", help="ขนาดของโมเดล (base, large, xl)")
    parser.add_argument("--task", type=str, help="ประเภทงาน (classification, summarization, etc.)")
    parser.add_argument("--input_file", type=str, help="ไฟล์ข้อมูลสำหรับเทรนโมเดล (.csv, .jsonl, etc.)")
    parser.add_argument("--output_dir", type=str, help="ไดเรกทอรีสำหรับบันทึกโมเดล")
    
    # Optional arguments
    parser.add_argument("--custom_model", type=str, help="โมเดลที่กำหนดเอง (ถ้าไม่ใช่โมเดลในรายการ)")
    parser.add_argument("--num_labels", type=int, help="จำนวน labels สำหรับงาน classification")
    parser.add_argument("--text_column", type=str, default="text", help="ชื่อคอลัมน์ข้อความ")
    parser.add_argument("--label_column", type=str, default="label", help="ชื่อคอลัมน์ label")
    parser.add_argument("--summary_column", type=str, help="ชื่อคอลัมน์ summary (สำหรับงาน summarization)")
    parser.add_argument("--source_column", type=str, help="ชื่อคอลัมน์ข้อความต้นทาง (สำหรับงาน translation)")
    parser.add_argument("--target_column", type=str, help="ชื่อคอลัมน์ข้อความปลายทาง (สำหรับงาน translation)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="จำนวน epochs")
    parser.add_argument("--max_length", type=int, default=512, help="ความยาวสูงสุดของข้อความ")
    parser.add_argument("--validation_split", type=float, default=0.2, help="สัดส่วนข้อมูลสำหรับ validation")
    parser.add_argument("--test_split", type=float, default=0.1, help="สัดส่วนข้อมูลสำหรับ test")
    parser.add_argument("--early_stopping", action="store_true", help="ใช้ early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="จำนวน epochs ที่ต้องรอก่อนหยุด")
    parser.add_argument("--fp16", action="store_true", help="ใช้ mixed precision training")
    
    # HF Hub parameters
    parser.add_argument("--push_to_hub", action="store_true", help="อัพโหลดโมเดลขึ้น Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="ชื่อโมเดลบน Hub (username/model-name)")
    parser.add_argument("--hub_token", type=str, help="Hugging Face token")
    
    args = parser.parse_args()
    
    # ถ้าใช้ --list_models หรือ --list_tasks
    if args.list_models:
        get_available_models()
        return
    
    if args.list_tasks:
        get_available_tasks()
        return
    
    # ตรวจสอบว่ามีพารามิเตอร์ที่จำเป็นครบไหม
    required_args = ["model", "task", "input_file", "output_dir"]
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    
    if missing_args and not (args.list_models or args.list_tasks or args.custom_model):
        if args.custom_model is None and "model" in missing_args:
            print(f"กรุณาระบุพารามิเตอร์ที่จำเป็น: {', '.join(missing_args)}")
            parser.print_help()
            return
    
    # แสดงข้อมูลการเทรน
    print("\n" + "=" * 50)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 50)
    
    if args.custom_model:
        print(f"Custom Model: {args.custom_model}")
    else:
        print(f"Model: {args.model}-{args.model_size}")
    
    print(f"Task: {args.task}")
    print(f"Input File: {args.input_file}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Maximum Length: {args.max_length}")
    print(f"Using Early Stopping: {args.early_stopping}")
    print(f"Using Mixed Precision: {args.fp16}")
    print("=" * 50 + "\n")
    
    # สร้าง fine-tuner
    fine_tuner = ModelFineTuner(
        model_name=args.model if not args.custom_model else None,
        model_size=args.model_size,
        task_type=args.task,
        num_labels=args.num_labels,
        max_length=args.max_length,
        custom_model_path=args.custom_model
    )
    
    # เตรียมข้อมูล
    print("Preparing dataset...")
    
    # เพิ่มพารามิเตอร์สำหรับงานเฉพาะ
    dataset_kwargs = {}
    if args.task == "summarization" and args.summary_column:
        dataset_kwargs["summary_column"] = args.summary_column
    
    if args.task == "translation":
        if args.source_column:
            dataset_kwargs["source_column"] = args.source_column
        if args.target_column:
            dataset_kwargs["target_column"] = args.target_column
    
    dataset = fine_tuner.prepare_dataset(
        data=args.input_file,
        text_column=args.text_column,
        label_column=args.label_column,
        train_test_split_ratio=args.test_split,
        validation_split=True,
        **dataset_kwargs
    )
    
    # แสดงข้อมูลจำนวนตัวอย่าง
    print("\nDataset Statistics:")
    for split, data in dataset.items():
        print(f"- {split}: {len(data)} samples")
    
    # เทรนโมเดล
    print("\nStarting training...")
    results = fine_tuner.train(
        dataset=dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token
    )
    
    # แสดงผลลัพธ์
    print("\nTraining completed!")
    print(f"Model saved to: {os.path.abspath(args.output_dir)}")
    
    # แสดงผลการประเมิน
    if results['test_results']:
        print("\nTest Results:")
        for metric, value in results['test_results'].items():
            print(f"- {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
