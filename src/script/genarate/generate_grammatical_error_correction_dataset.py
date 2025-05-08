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

# Types of errors in Thai
ERROR_TYPES = {
    'WORD_ORDER': 'การเรียงลำดับคำผิด เช่น "ฉันชอบมากหนังสือเล่มนี้" แทนที่จะเป็น "ฉันชอบหนังสือเล่มนี้มาก"',
    'MISSING_WORD': 'การขาดคำที่จำเป็น เช่น "ฉันไปโรงเรียน" แทนที่จะเป็น "ฉันไปที่โรงเรียน"',
    'EXTRA_WORD': 'การใช้คำเกินความจำเป็น เช่น "ฉันกำลังจะไปที่ไปบ้าน" แทนที่จะเป็น "ฉันกำลังจะไปบ้าน"',
    'WRONG_WORD_CHOICE': 'การเลือกใช้คำผิด เช่น "เขากำลังกินการบ้าน" แทนที่จะเป็น "เขากำลังทำการบ้าน"',
    'CLASSIFIER_ERROR': 'การใช้ลักษณนามผิด เช่น "แมวสองตัวใบ" แทนที่จะเป็น "แมวสองตัว"',
    'REGISTER_ERROR': 'การใช้ระดับภาษาผิด เช่น "ผมขอถามอาจารย์ว่ามึงจะไปไหน" แทนที่จะเป็น "ผมขอถามอาจารย์ว่าท่านจะไปไหน"',
    'TONE_ERROR': 'การใช้ผิดวรรณยุกต์ เช่น "เขามีแม่ว" แทนที่จะเป็น "เขามีแม่ว่า"',
    'SPACING_ERROR': 'การเว้นวรรคผิด เช่น "ฉันชอบกิน ข้าว" แทนที่จะเป็น "ฉันชอบกินข้าว"',
    'CONJUNCTION_ERROR': 'การใช้คำเชื่อมผิด เช่น "เขาเรียนเก่งกับขยัน" แทนที่จะเป็น "เขาเรียนเก่งและขยัน"',
    'REDUNDANCY': 'การใช้คำซ้ำซ้อน เช่น "ทุกๆคนทั้งหมด" แทนที่จะเป็น "ทุกคน"'
}

# Content domains to generate examples from
DOMAINS = [
    'บทความวิชาการ', 'บทความทั่วไป', 'จดหมาย', 'อีเมล', 'โพสต์บนโซเชียลมีเดีย',
    'ข้อความส่วนตัว', 'การบ้านนักเรียน', 'ข่าว', 'คำบรรยายภาพ', 'บทสนทนา'
]

def create_gec_prompt(domain, num_examples=5):
    """
    Create a prompt for generating Thai text with grammatical errors and corrections
    
    Args:
        domain (str): Domain of the content
        num_examples (int): Number of examples to generate
        
    Returns:
        str: The generated prompt
    """
    error_description = "\n".join([f"- {error_type}: {description}" for error_type, description in ERROR_TYPES.items()])
    
    prompt = f"""สร้างตัวอย่างประโยคภาษาไทยที่มีข้อผิดพลาดทางไวยากรณ์ในบริบทเกี่ยวกับ{domain} พร้อมทั้งระบุประเภทข้อผิดพลาดและการแก้ไขที่ถูกต้อง

ประเภทข้อผิดพลาดทางไวยากรณ์ที่ต้องการ:
{error_description}

กรุณาสร้างตัวอย่างจำนวน {num_examples} ตัวอย่าง โดยแต่ละตัวอย่างเป็นประโยคที่มีข้อผิดพลาดทางไวยากรณ์ประเภทที่แตกต่างกัน

กรุณาตอบในรูปแบบนี้:

ตัวอย่างที่ 1:
ประโยคที่มีข้อผิดพลาด: [ประโยคที่มีข้อผิดพลาดทางไวยากรณ์]
ประเภทข้อผิดพลาด: [ประเภทข้อผิดพลาด]
คำอธิบาย: [อธิบายว่าข้อผิดพลาดคืออะไรและเพราะอะไรจึงผิด]
ประโยคที่ถูกต้อง: [ประโยคที่แก้ไขแล้ว]

ตัวอย่างที่ 2:
...

พยายามสร้างตัวอย่างที่หลากหลายและเป็นธรรมชาติ ครอบคลุมประเภทข้อผิดพลาดที่แตกต่างกัน"""
    
    return prompt

def generate_gec_samples(api_key, num_samples, domains=None):
    """
    Generate grammatical error correction samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        num_samples (int): Number of samples to generate
        domains (list): List of domains to generate samples from
        
    Returns:
        list: Generated samples
    """
    if domains is None:
        domains = DOMAINS
    
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในการวิเคราะห์และแก้ไขข้อผิดพลาดทางไวยากรณ์ภาษาไทย คุณเข้าใจกฎเกณฑ์และหลักการใช้ภาษาไทยอย่างถูกต้อง"""
    
    results = []
    samples_needed = num_samples
    
    # Calculate samples per API call based on desired examples per prompt
    examples_per_prompt = 5
    samples_per_api_call = min(examples_per_prompt, samples_needed)
    
    # Create a distribution of domains
    domain_distribution = []
    while len(domain_distribution) < num_samples:
        random.shuffle(domains)
        domain_distribution.extend(domains)
    domain_distribution = domain_distribution[:num_samples]
    
    # Group by domains to make fewer API calls
    domain_groups = {}
    for i, domain in enumerate(domain_distribution):
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(i)
    
    # For each domain group
    for domain, _ in tqdm(domain_groups.items(), desc="Generating GEC samples by domain"):
        prompt = create_gec_prompt(domain, samples_per_api_call)
        
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
                        # Split the response by example markers
                        examples = response.split("ตัวอย่างที่ ")
                        
                        # Skip the first split which might be empty or contain introduction
                        for i, example in enumerate(examples[1:], start=1):
                            # Skip if we've already collected enough samples
                            if len(results) >= num_samples:
                                break
                                
                            # Check if the example has all required parts
                            if "ประโยคที่มีข้อผิดพลาด:" not in example or "ประโยคที่ถูกต้อง:" not in example:
                                continue
                            
                            # Extract error sentence
                            error_sentence_marker = "ประโยคที่มีข้อผิดพลาด:"
                            error_type_marker = "ประเภทข้อผิดพลาด:"
                            explanation_marker = "คำอธิบาย:"
                            correct_sentence_marker = "ประโยคที่ถูกต้อง:"
                            
                            # Extract error sentence
                            if error_sentence_marker in example and error_type_marker in example:
                                error_sentence = example.split(error_sentence_marker)[1].split(error_type_marker)[0].strip()
                            else:
                                continue
                            
                            # Extract error type
                            if error_type_marker in example and explanation_marker in example:
                                error_type = example.split(error_type_marker)[1].split(explanation_marker)[0].strip()
                            elif error_type_marker in example and correct_sentence_marker in example:
                                error_type = example.split(error_type_marker)[1].split(correct_sentence_marker)[0].strip()
                                if "คำอธิบาย:" in error_type:
                                    error_type = error_type.split("คำอธิบาย:")[0].strip()
                            else:
                                continue
                            
                            # Extract explanation
                            explanation = ""
                            if explanation_marker in example and correct_sentence_marker in example:
                                explanation = example.split(explanation_marker)[1].split(correct_sentence_marker)[0].strip()
                            
                            # Extract correct sentence
                            if correct_sentence_marker in example:
                                correct_sentence_parts = example.split(correct_sentence_marker)[1].split("ตัวอย่างที่ ")
                                correct_sentence = correct_sentence_parts[0].strip()
                                # If there's a next example number in the correct sentence, trim it
                                if len(correct_sentence_parts) > 1 and correct_sentence_parts[1].strip().isdigit():
                                    next_example_num = int(correct_sentence_parts[1].strip())
                                    if next_example_num == i + 1:
                                        correct_sentence = correct_sentence.rsplit("\n", 1)[0].strip()
                            else:
                                continue
                            
                            # Create sample
                            sample = {
                                "id": len(results) + 1,
                                "domain": domain,
                                "error_sentence": error_sentence,
                                "error_type": error_type,
                                "explanation": explanation,
                                "correct_sentence": correct_sentence,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Only add if both sentences are different (there's an actual error)
                            if error_sentence != correct_sentence:
                                results.append(sample)
                        
                        # If we got at least one sample from this response, consider it a success
                        if any(sample['domain'] == domain for sample in results):
                            break
                        else:
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_DELAY)
                            else:
                                print(f"Failed to extract valid samples from response for domain: {domain}")
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
    parser = argparse.ArgumentParser(description='Generate Thai Grammatical Error Correction dataset using Deepseek API')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_gec_dataset_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=50,
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
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_gec_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_gec_samples(api_key, args.samples, args.domains)
    
    # Save samples
    save_samples(samples, args.output_file)

if __name__ == "__main__":
    main()