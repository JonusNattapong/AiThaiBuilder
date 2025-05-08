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

# Define semantic roles in Thai
SEMANTIC_ROLES = {
    'AGENT': 'ผู้กระทำ - ผู้ที่ทำกริยา',
    'PATIENT': 'ผู้ถูกกระทำ - ผู้ที่ได้รับผลจากการกระทำ',
    'EXPERIENCER': 'ผู้ประสบ - ผู้ที่รู้สึกหรือประสบเหตุการณ์',
    'THEME': 'ตัวกระทำเหตุการณ์ - สิ่งที่ถูกเคลื่อนย้ายหรือเปลี่ยนแปลง',
    'INSTRUMENT': 'เครื่องมือ - สิ่งที่ใช้ในการกระทำ',
    'LOCATION': 'สถานที่ - ที่ที่เกิดเหตุการณ์',
    'DESTINATION': 'จุดหมาย - สถานที่ที่มีการเคลื่อนที่ไปถึง',
    'SOURCE': 'ต้นทาง - สถานที่เริ่มต้นของการเคลื่อนที่',
    'TIME': 'เวลา - เวลาที่เกิดเหตุการณ์',
    'MANNER': 'ลักษณะ - วิธีการที่กระทำ',
    'PURPOSE': 'จุดประสงค์ - เป้าหมายของการกระทำ',
    'CAUSE': 'สาเหตุ - สิ่งที่ทำให้เกิดเหตุการณ์',
    'BENEFICIARY': 'ผู้ได้รับประโยชน์ - ผู้ที่ได้รับประโยชน์จากเหตุการณ์',
    'QUANTITY': 'ปริมาณ - จำนวนหรือปริมาณที่ระบุในประโยค',
    'PREDICATE': 'ภาคแสดงหรือกริยา - การกระทำหรือเหตุการณ์ในประโยค'
}

# Content domains to generate examples from
DOMAINS = [
    'ข่าว', 'บทความวิชาการ', 'บทความทั่วไป', 'กีฬา', 'บันเทิง', 
    'เทคโนโลยี', 'ธุรกิจ', 'การศึกษา', 'สุขภาพ', 'การเมือง'
]

def create_srl_prompt(domain, num_samples=3):
    """
    Create a prompt for generating Thai text with semantic role labeling
    
    Args:
        domain (str): Domain of the content
        num_samples (int): Number of samples to generate
        
    Returns:
        str: The generated prompt
    """
    roles_description = "\n".join([f"- {role}: {description}" for role, description in SEMANTIC_ROLES.items()])
    
    prompt = f"""สร้างตัวอย่างประโยคภาษาไทยในหัวข้อเกี่ยวกับ{domain} พร้อมระบุบทบาททางความหมาย (Semantic Roles) ของคำและวลีในประโยค

บทบาททางความหมายแต่ละประเภทมีคำอธิบายดังนี้:
{roles_description}

กรุณาสร้างประโยคจำนวน {num_samples} ประโยค ที่มีความซับซ้อนเพียงพอให้มีบทบาททางความหมายหลากหลาย โดยแต่ละประโยคควรมีบทบาททางความหมายอย่างน้อย 4 บทบาท

กรุณาตอบในรูปแบบนี้สำหรับแต่ละประโยค:

ประโยค: [ประโยคภาษาไทย]
การวิเคราะห์:
1. [คำหรือวลี] - [บทบาท]: [คำอธิบายเหตุผล]
2. [คำหรือวลี] - [บทบาท]: [คำอธิบายเหตุผล]
...

โดยการวิเคราะห์ให้เรียงลำดับบทบาทตามตำแหน่งที่ปรากฏในประโยค"""
    
    return prompt

def generate_srl_samples(api_key, num_samples, domains=None):
    """
    Generate semantic role labeling samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        num_samples (int): Number of samples to generate
        domains (list): List of domains to generate samples from
        
    Returns:
        list: Generated samples
    """
    if domains is None:
        domains = DOMAINS
    
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในการวิเคราะห์บทบาททางความหมาย (Semantic Role Labeling) ในภาษาไทย คุณมีความเข้าใจโครงสร้างประโยคและสามารถระบุบทบาทของคำและวลีในประโยคได้อย่างถูกต้อง"""
    
    results = []
    
    # For each domain
    domains_per_sample = max(1, num_samples // len(domains))
    extra_samples = num_samples % len(domains)
    
    # Create a list of domains to generate samples from
    generation_list = domains * domains_per_sample
    generation_list.extend(random.sample(domains, extra_samples))
    
    # Shuffle the list to get a good mix of domains
    random.shuffle(generation_list)
    
    for domain in tqdm(generation_list, desc="Generating SRL samples"):
        # Generate 3 samples per API call to reduce the number of API calls
        samples_per_call = 3
        prompt = create_srl_prompt(domain, samples_per_call)
        
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
                        # Split the response by "ประโยค:" to get individual samples
                        samples_text = response.split("ประโยค:")
                        
                        # Skip the first split which is typically empty or contains introduction
                        for sample_text in samples_text[1:]:
                            # Check if there's a valid sample
                            if "การวิเคราะห์:" not in sample_text:
                                continue
                            
                            # Extract the sentence and analysis
                            sentence = sample_text.split("การวิเคราะห์:")[0].strip()
                            analysis_text = sample_text.split("การวิเคราะห์:")[1].strip()
                            
                            # Parse the analysis
                            roles = []
                            for line in analysis_text.split('\n'):
                                if not line.strip() or ":" not in line:
                                    continue
                                
                                # Try to parse each role line
                                try:
                                    # Extract index number if exists
                                    if line[0].isdigit() and '. ' in line:
                                        line = line.split('. ', 1)[1]
                                    
                                    # Split the line at the first hyphen that separates the phrase and role
                                    phrase_role_parts = line.split(' - ', 1)
                                    if len(phrase_role_parts) < 2:
                                        continue
                                    
                                    phrase, role_explanation = phrase_role_parts
                                    
                                    # Extract role and explanation
                                    role_parts = role_explanation.split(':', 1)
                                    if len(role_parts) < 2:
                                        role = role_parts[0].strip()
                                        explanation = ""
                                    else:
                                        role, explanation = role_parts
                                    
                                    roles.append({
                                        "phrase": phrase.strip(),
                                        "role": role.strip(),
                                        "explanation": explanation.strip() if explanation else ""
                                    })
                                except Exception as e:
                                    print(f"Error parsing role line '{line}': {e}")
                            
                            # Create sample
                            if roles:
                                sample = {
                                    "sentence": sentence,
                                    "domain": domain,
                                    "roles": roles,
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
    parser = argparse.ArgumentParser(description='Generate Thai Semantic Role Labeling dataset using Deepseek API')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_srl_dataset_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=30,
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
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_srl_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_srl_samples(api_key, args.samples, args.domains)
    
    # Save samples
    save_samples(samples, args.output_file)

if __name__ == "__main__":
    main()