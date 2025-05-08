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

# Entity types definition
ENTITY_TYPES = {
    'PERSON': 'บุคคล (ชื่อคน)',
    'ORGANIZATION': 'องค์กร (ชื่อบริษัท, หน่วยงาน, องค์กร)',
    'LOCATION': 'สถานที่ (ชื่อเมือง, ประเทศ, สถานที่)',
    'DATE': 'วันที่ (วัน เดือน ปี)',
    'TIME': 'เวลา',
    'MONEY': 'จำนวนเงิน',
    'PERCENT': 'เปอร์เซ็นต์',
    'PRODUCT': 'ผลิตภัณฑ์ (ชื่อสินค้า)',
    'EVENT': 'เหตุการณ์ (ชื่องาน, เทศกาล)',
    'LAW': 'กฎหมาย (ชื่อกฎหมาย)',
    'LANGUAGE': 'ภาษา (ชื่อภาษา)',
}

# Domains to generate examples from
DOMAINS = [
    'ข่าว', 'การเมือง', 'กีฬา', 'บันเทิง', 'เทคโนโลยี', 
    'ธุรกิจ', 'การศึกษา', 'สุขภาพ', 'สิ่งแวดล้อม', 'ท่องเที่ยว'
]

def create_ner_prompt(domain, entities_to_include=None):
    """
    Create a prompt for generating Thai text with named entities
    
    Args:
        domain (str): Domain to generate text from
        entities_to_include (list): List of entity types to include
        
    Returns:
        str: The generated prompt
    """
    if entities_to_include is None:
        # Randomly select 3-5 entity types to include
        num_entities = random.randint(3, 5)
        entities_to_include = random.sample(list(ENTITY_TYPES.keys()), num_entities)
    
    entities_description = '\n'.join([f"- {entity}: {ENTITY_TYPES[entity]}" for entity in entities_to_include])
    
    prompt = f"""สร้างข้อความภาษาไทยในหัวข้อเกี่ยวกับ "{domain}" ที่มีการใช้ชื่อเฉพาะ (Named Entities) ต่อไปนี้:

{entities_description}

กรุณาสร้างข้อความที่เป็นธรรมชาติ ความยาว 3-5 ประโยค และมีการใช้ชื่อเฉพาะที่ระบุข้างต้นให้มากที่สุด

หลังจากนั้น ให้ระบุชื่อเฉพาะที่ปรากฏในข้อความ ในรูปแบบต่อไปนี้:
ชื่อเฉพาะ: [คำหรือวลี] - ประเภท: [ประเภทชื่อเฉพาะ]

ตัวอย่างเช่น:
ชื่อเฉพาะ: เวโลโดรมหัวหมาก - ประเภท: LOCATION
ชื่อเฉพาะ: นายวิษณุ เครืองาม - ประเภท: PERSON

กรุณาตอบในรูปแบบนี้:

ข้อความ:
[ข้อความที่สร้าง]

ชื่อเฉพาะที่พบ:
[รายการชื่อเฉพาะ]"""
    
    return prompt

def generate_ner_samples(api_key, num_samples, domains=None):
    """
    Generate NER samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        num_samples (int): Number of samples to generate
        domains (list): List of domains to generate samples from
        
    Returns:
        list: Generated samples
    """
    if domains is None:
        domains = DOMAINS
    
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่มีชื่อเฉพาะ (Named Entities) ตามที่กำหนด กรุณาสร้างข้อความที่เป็นธรรมชาติและระบุชื่อเฉพาะที่ปรากฏในข้อความให้ถูกต้อง"""
    
    results = []
    
    for _ in tqdm(range(num_samples), desc="Generating NER samples"):
        domain = random.choice(domains)
        
        # Randomly decide how many entity types to include
        num_entity_types = random.randint(3, 5)
        entity_types = random.sample(list(ENTITY_TYPES.keys()), num_entity_types)
        
        prompt = create_ner_prompt(domain, entity_types)
        
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
                        # Extract the text and entities
                        text_part = response.split("ข้อความ:")[1].split("ชื่อเฉพาะที่พบ:")[0].strip()
                        entities_part = response.split("ชื่อเฉพาะที่พบ:")[1].strip()
                        
                        # Parse entities
                        entities = []
                        for line in entities_part.split('\n'):
                            if line.strip() and "ชื่อเฉพาะ:" in line and "ประเภท:" in line:
                                entity_text = line.split("ชื่อเฉพาะ:")[1].split("ประเภท:")[0].strip()
                                entity_type = line.split("ประเภท:")[1].strip()
                                entities.append({"text": entity_text, "type": entity_type})
                        
                        # Create sample
                        sample = {
                            "text": text_part,
                            "entities": entities,
                            "domain": domain,
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
    parser = argparse.ArgumentParser(description='Generate Thai NER dataset using Deepseek API')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_ner_dataset_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to generate')
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
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_ner_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_ner_samples(api_key, args.samples, args.domains)
    
    # Save samples
    save_samples(samples, args.output_file)

if __name__ == "__main__":
    main()