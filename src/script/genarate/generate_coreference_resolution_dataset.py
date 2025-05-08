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

# Types of coreference
COREFERENCE_TYPES = {
    'PRONOUN': 'การอ้างถึงด้วยคำสรรพนาม เช่น เขา, เธอ, มัน, ผม, ฉัน, พวกเขา, พวกเรา, ที่',
    'DEMONSTRATIVE': 'การอ้างถึงด้วยคำชี้เฉพาะ เช่น นี่, นั่น, โน่น, อันนี้, คนนั้น, อันนั้น, เรื่องนี้',
    'NAME_ALIAS': 'การอ้างถึงด้วยชื่อเล่น ชื่อเต็ม หรือการเรียกแบบอื่น เช่น คุณหมอ (หมอกานต์), นายก (นายกรัฐมนตรี)',
    'ZERO': 'การละคำที่อ้างถึง ซึ่งเป็นลักษณะพิเศษในภาษาไทย เช่น "แม่บอกว่า(แม่)จะกลับบ้านเย็นนี้"',
    'DEFINITE_NP': 'การอ้างถึงด้วยวลีนามที่เจาะจง เช่น "นักเรียนคนเก่ง" ที่อ้างถึง "เด็กหญิงสมศรี"',
    'REPETITION': 'การอ้างถึงด้วยการกล่าวซ้ำอีกครั้ง'
}

# Content domains to generate examples from
DOMAINS = [
    'ข่าว', 'นิทาน', 'บทความวิชาการ', 'บทความทั่วไป', 'กีฬา', 
    'บันเทิง', 'บทสนทนา', 'ประวัติศาสตร์', 'นวนิยาย', 'วิทยาศาสตร์'
]

def create_coref_prompt(domain, num_paragraphs=2):
    """
    Create a prompt for generating Thai text with coreference examples
    
    Args:
        domain (str): Domain of the content
        num_paragraphs (int): Number of paragraphs to generate
        
    Returns:
        str: The generated prompt
    """
    types_description = "\n".join([f"- {coref_type}: {description}" for coref_type, description in COREFERENCE_TYPES.items()])
    
    prompt = f"""สร้างตัวอย่างข้อความภาษาไทยในหัวข้อเกี่ยวกับ{domain} ที่มีการอ้างอิงข้ามประโยค (Coreference) ในหลากหลายรูปแบบ

รูปแบบการอ้างอิงที่ควรรวมอยู่ในข้อความ:
{types_description}

กรุณาสร้างข้อความที่ประกอบด้วย {num_paragraphs} ย่อหน้า โดยในแต่ละย่อหน้าควรมีประโยคอย่างน้อย 3 ประโยค และมีการอ้างอิงข้ามประโยคอย่างน้อย 4 รูปแบบ

กรุณาตอบในรูปแบบนี้:

ข้อความ:
[ข้อความภาษาไทย โดยใส่หมายเลขกำกับแต่ละประโยค เช่น [1], [2], ... เพื่อความสะดวกในการอ้างอิง]

การวิเคราะห์:
1. "[คำหรือวลีที่ถูกอ้างถึง]" ในประโยค [หมายเลขประโยค] อ้างถึง "[คำหรือวลีต้นทาง]" ในประโยค [หมายเลขประโยค] ประเภท: [ประเภทการอ้างอิง]
2. "[คำหรือวลีที่ถูกอ้างถึง]" ในประโยค [หมายเลขประโยค] อ้างถึง "[คำหรือวลีต้นทาง]" ในประโยค [หมายเลขประโยค] ประเภท: [ประเภทการอ้างอิง]
...

ให้เรียงลำดับการวิเคราะห์ตามหมายเลขประโยคที่มีคำที่ถูกอ้างถึง"""
    
    return prompt

def generate_coref_samples(api_key, num_samples, domains=None):
    """
    Generate coreference resolution samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        num_samples (int): Number of samples to generate
        domains (list): List of domains to generate samples from
        
    Returns:
        list: Generated samples
    """
    if domains is None:
        domains = DOMAINS
    
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในการวิเคราะห์การอ้างอิงข้ามประโยค (Coreference Resolution) ในภาษาไทย คุณสามารถระบุคำหรือวลีที่อ้างถึงกันในข้อความได้อย่างถูกต้อง"""
    
    results = []
    
    # Determine how many samples to generate per domain
    samples_per_domain = max(1, num_samples // len(domains))
    extra_samples = num_samples % len(domains)
    
    # Create a list of domains to generate samples from
    domain_list = domains * samples_per_domain
    domain_list.extend(random.sample(domains, extra_samples))
    
    # Shuffle the list to avoid consecutive samples from the same domain
    random.shuffle(domain_list)
    
    for domain in tqdm(domain_list, desc="Generating coreference samples"):
        prompt = create_coref_prompt(domain)
        
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
                        # Extract text and analysis
                        text_marker = "ข้อความ:"
                        analysis_marker = "การวิเคราะห์:"
                        
                        if text_marker in response and analysis_marker in response:
                            text_part = response.split(text_marker)[1].split(analysis_marker)[0].strip()
                            analysis_part = response.split(analysis_marker)[1].strip()
                            
                            # Parse coreference mentions
                            mentions = []
                            for line in analysis_part.split('\n'):
                                if not line.strip() or "อ้างถึง" not in line or "ประเภท:" not in line:
                                    continue
                                
                                try:
                                    # Remove the index number if it exists
                                    if line[0].isdigit() and '. ' in line:
                                        line = line.split('. ', 1)[1]
                                    
                                    # Extract mention, referent, sentences, and type
                                    mention_part = line.split('" ในประโยค ')[0].strip('"')
                                    if 'อ้างถึง "' not in line:
                                        continue
                                    
                                    rest_part = line.split('อ้างถึง "')[1]
                                    referent = rest_part.split('" ในประโยค ')[0].strip()
                                    
                                    sentences_type_part = rest_part.split('" ในประโยค ')[1]
                                    mention_sentence = sentences_type_part.split(' อ้างถึง "')[0].strip().strip('[]')
                                    
                                    if 'ประเภท:' not in sentences_type_part:
                                        continue
                                    
                                    type_part = sentences_type_part.split('ประเภท:')[1].strip()
                                    coref_type = type_part.split()[0].strip() if type_part else "UNKNOWN"
                                    
                                    # Extract referent sentence number - check various patterns
                                    referent_sentence = None
                                    if 'ในประโยค [' in sentences_type_part:
                                        referent_sentence_part = sentences_type_part.split('ในประโยค [')[1].split(']')[0]
                                        referent_sentence = referent_sentence_part
                                    
                                    # If we couldn't extract the referent sentence, skip this mention
                                    if not referent_sentence:
                                        continue
                                    
                                    mentions.append({
                                        "mention": mention_part,
                                        "referent": referent,
                                        "mention_sentence": mention_sentence,
                                        "referent_sentence": referent_sentence,
                                        "type": coref_type
                                    })
                                except Exception as e:
                                    print(f"Error parsing mention line '{line}': {e}")
                            
                            # Create sample if we have valid mentions
                            if mentions:
                                sample = {
                                    "text": text_part,
                                    "domain": domain,
                                    "coreference_mentions": mentions,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                results.append(sample)
                                break
                        else:
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_DELAY)
                            else:
                                print("Failed to find text and analysis markers in response")
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
    
    # Ensure we have the exact number of samples requested
    return results[:num_samples]

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
    parser = argparse.ArgumentParser(description='Generate Thai Coreference Resolution dataset using Deepseek API')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_coref_dataset_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=20,
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
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_coref_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_coref_samples(api_key, args.samples, args.domains)
    
    # Save samples
    save_samples(samples, args.output_file)

if __name__ == "__main__":
    main()