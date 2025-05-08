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

# Text categorties to generate summaries for
TEXT_CATEGORIES = [
    'ข่าว', 'บทความวิชาการ', 'บทความทั่วไป', 'บทความวิจัย', 
    'บทความวิเคราะห์', 'บทสัมภาษณ์', 'รีวิว', 'ประกาศ'
]

# Domains to generate content from
DOMAINS = [
    'การเมือง', 'เศรษฐกิจ', 'สังคม', 'กีฬา', 'บันเทิง', 
    'เทคโนโลยี', 'สุขภาพ', 'การศึกษา', 'สิ่งแวดล้อม', 'วัฒนธรรม'
]

# Summarization types
SUMMARY_TYPES = [
    'extractive', # ดึงประโยคสำคัญจากต้นฉบับ
    'abstractive' # สร้างประโยคใหม่ที่สรุปใจความสำคัญ
]

def create_summarization_prompt(category, domain, min_length=300, max_length=800):
    """
    Create a prompt for generating Thai text with summary
    
    Args:
        category (str): Text category
        domain (str): Domain to generate text from
        min_length (int): Minimum text length
        max_length (int): Maximum text length
        
    Returns:
        str: The generated prompt
    """
    summary_type = random.choice(SUMMARY_TYPES)
    summary_type_text = "เชิงสรุปข้อความสำคัญและเรียบเรียงใหม่" if summary_type == "abstractive" else "เชิงดึงประโยคสำคัญจากต้นฉบับ"
    
    prompt = f"""สร้าง{category}ภาษาไทยเกี่ยวกับ{domain} ความยาวประมาณ {min_length}-{max_length} คำ พร้อมกับสรุปความ

กรุณาสร้างเนื้อหาที่มีความน่าสนใจ มีข้อมูลที่หลากหลาย และสมจริง

หลังจากนั้น ให้เขียนบทสรุป{summary_type_text} (ความยาวประมาณ 15-20% ของเนื้อหาต้นฉบับ)

กรุณาตอบในรูปแบบนี้:

เนื้อหาต้นฉบับ:
[เนื้อหาที่สร้าง]

บทสรุป ({summary_type_text}):
[บทสรุปของเนื้อหา]"""
    
    return prompt, summary_type

def generate_summarization_samples(api_key, num_samples, categories=None, domains=None):
    """
    Generate summarization samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        num_samples (int): Number of samples to generate
        categories (list): List of text categories
        domains (list): List of domains to generate samples from
        
    Returns:
        list: Generated samples
    """
    if categories is None:
        categories = TEXT_CATEGORIES
    
    if domains is None:
        domains = DOMAINS
    
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในการสร้างเนื้อหาภาษาไทยและการสรุปความ โดยสามารถสร้างเนื้อหาได้หลากหลายรูปแบบและสรุปใจความสำคัญได้อย่างมีประสิทธิภาพ"""
    
    results = []
    
    for _ in tqdm(range(num_samples), desc="Generating summarization samples"):
        category = random.choice(categories)
        domain = random.choice(domains)
        
        # Randomize text length requirement
        min_length = random.randint(300, 500)
        max_length = random.randint(600, 1000)
        
        prompt, summary_type = create_summarization_prompt(category, domain, min_length, max_length)
        
        # Retry mechanism
        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = generate_with_deepseek(messages, api_key, max_tokens=1500)
                
                if response and not response.startswith("Error:"):
                    # Parse the response
                    try:
                        # Extract the text and summary
                        text_part = response.split("เนื้อหาต้นฉบับ:")[1].split("บทสรุป")[0].strip()
                        summary_part = response.split("บทสรุป")[1].split(":", 1)[1].strip()
                        
                        # Create sample
                        sample = {
                            "document": text_part,
                            "summary": summary_part,
                            "category": category,
                            "domain": domain,
                            "summary_type": summary_type,
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
    parser = argparse.ArgumentParser(description='Generate Thai text summarization dataset using Deepseek API')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_summarization_dataset_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Specific text categories to generate (default: all)')
    parser.add_argument('--domains', type=str, nargs='+', default=None,
                        help='Specific domains to generate (default: all)')
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
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_summarization_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_summarization_samples(api_key, args.samples, args.categories, args.domains)
    
    # Save samples
    save_samples(samples, args.output_file)

if __name__ == "__main__":
    main()