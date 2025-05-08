import os
import json
import random
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Deepseek utilities
from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

# Load environment variables
load_dotenv()

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1
OUTPUT_DIR = 'output'

# Classification tasks and their labels
CLASSIFICATION_TASKS = {
    'news_category': {
        'description': 'หมวดหมู่ข่าว',
        'labels': [
            'การเมือง', 'เศรษฐกิจ', 'สังคม', 'กีฬา', 'บันเทิง', 
            'เทคโนโลยี', 'สุขภาพ', 'การศึกษา', 'สิ่งแวดล้อม', 'อาชญากรรม'
        ]
    },
    'product_review': {
        'description': 'รีวิวสินค้า',
        'labels': [
            'อิเล็กทรอนิกส์', 'เครื่องสำอาง', 'เสื้อผ้า', 'อาหารและเครื่องดื่ม', 
            'เครื่องใช้ในบ้าน', 'ของเล่น', 'หนังสือ', 'อุปกรณ์กีฬา', 'ยานพาหนะ', 'เฟอร์นิเจอร์'
        ]
    },
    'sentiment': {
        'description': 'ความรู้สึก',
        'labels': [
            'เชิงบวก', 'เชิงลบ', 'เป็นกลาง'
        ]
    },
    'intent': {
        'description': 'เจตนาในการสื่อสาร',
        'labels': [
            'สอบถามข้อมูล', 'ร้องเรียน', 'ชมเชย', 'ขอคำแนะนำ', 
            'แจ้งปัญหา', 'ขอความช่วยเหลือ', 'แสดงความคิดเห็น', 'สั่งซื้อสินค้า'
        ]
    },
    'content_type': {
        'description': 'ประเภทเนื้อหา',
        'labels': [
            'ข้อเท็จจริง', 'ความคิดเห็น', 'โฆษณา', 'ข่าวปลอม', 
            'เนื้อหาทางวิชาการ', 'เนื้อหาบันเทิง', 'เรื่องแต่ง', 'ประกาศ'
        ]
    },
    'toxicity': {
        'description': 'ความเป็นพิษในการสื่อสาร',
        'labels': [
            'ปกติ', 'ดูหมิ่น', 'ก้าวร้าว', 'คุกคาม', 'ลามก', 'เหยียดเชื้อชาติ'
        ]
    }
}

def create_classification_prompt(task, label, length_range=(50, 150)):
    """
    Create a prompt for generating text with specific classification label
    
    Args:
        task (str): Classification task
        label (str): Classification label
        length_range (tuple): Range of text length (min, max)
        
    Returns:
        str: The generated prompt
    """
    task_info = CLASSIFICATION_TASKS[task]
    task_description = task_info['description']
    min_length, max_length = length_range
    
    prompt = f"""สร้างข้อความภาษาไทยที่มีลักษณะเป็น "{label}" ในหมวดหมู่ "{task_description}"

ความยาวของข้อความควรอยู่ระหว่าง {min_length} ถึง {max_length} คำ
ข้อความควรมีความเป็นธรรมชาติ สมจริง และสอดคล้องกับป้ายกำกับที่กำหนดให้อย่างชัดเจน

กรุณาตอบในรูปแบบนี้:

ข้อความ:
[ข้อความที่สร้าง]"""
    
    return prompt

def generate_classification_samples(api_key, task, num_samples_per_label, length_range=(50, 150)):
    """
    Generate classification samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        task (str): Classification task
        num_samples_per_label (int): Number of samples to generate per label
        length_range (tuple): Range of text length (min, max)
        
    Returns:
        list: Generated samples
    """
    task_info = CLASSIFICATION_TASKS[task]
    labels = task_info['labels']
    task_description = task_info['description']
    
    system_prompt = f"""คุณเป็น AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยตามหมวดหมู่ "{task_description}" ที่กำหนด กรุณาสร้างข้อความที่มีลักษณะตรงตามป้ายกำกับที่ระบุอย่างชัดเจน"""
    
    results = []
    
    for label in tqdm(labels, desc=f"Generating {task} samples", leave=False):
        for _ in tqdm(range(num_samples_per_label), desc=f"Label: {label}", leave=False):
            prompt = create_classification_prompt(task, label, length_range)
            
            # Retry mechanism
            for attempt in range(MAX_RETRIES):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = generate_with_deepseek(messages, api_key)
                    
                    if response and not response.startswith("Error:"):
                        # Parse the response
                        try:
                            # Extract the text
                            text_part = response.split("ข้อความ:")[1].strip() if "ข้อความ:" in response else response.strip()
                            
                            # Create sample
                            sample = {
                                "text": text_part,
                                "label": label,
                                "task": task,
                                "task_description": task_description,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            results.append(sample)
                            break
                        except Exception as e:
                            print(f"Error parsing response: {e}")
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_DELAY)
                            else:
                                print(f"Failed to parse after {MAX_RETRIES} attempts")
                    else:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"Failed to generate after {MAX_RETRIES} attempts: {response}")
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"Exception: {str(e)}")
    
    return results

def save_samples(samples, output_file):
    """
    Save samples to JSON file
    
    Args:
        samples (list): Samples to save
        output_file (str): Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Thai text classification dataset using Deepseek API')
    parser.add_argument('--task', type=str, required=True, choices=list(CLASSIFICATION_TASKS.keys()),
                        help=f'Classification task: {", ".join(CLASSIFICATION_TASKS.keys())}')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_classification_[task]_[timestamp].json)')
    parser.add_argument('--samples-per-label', type=int, default=10,
                        help='Number of samples per label')
    parser.add_argument('--min-length', type=int, default=50,
                        help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=150,
                        help='Maximum text length')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Deepseek API key (will use environment variable if not provided)')
    
    args = parser.parse_args()
    
    # Load API key
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: Deepseek API key not found. Please provide it with --api-key or set DEEPSEEK_API_KEY in your .env file.")
        return
    
    # Set default output file if not provided
    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_classification_{args.task}_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_classification_samples(
        api_key, 
        args.task, 
        args.samples_per_label, 
        (args.min_length, args.max_length)
    )
    
    # Save samples
    save_samples(samples, args.output_file)
    
    # Print stats
    num_labels = len(CLASSIFICATION_TASKS[args.task]['labels'])
    target_samples = num_labels * args.samples_per_label
    print(f"Generated {len(samples)}/{target_samples} samples for task '{args.task}'")

if __name__ == "__main__":
    main()