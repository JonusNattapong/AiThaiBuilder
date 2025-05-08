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

# Words with multiple meanings in Thai
POLYSEMOUS_WORDS = {
    'แพง': ['มีราคาสูง', 'สัตว์ประเภทหนึ่ง คล้ายกวาง', 'พักพิง'],
    'ตา': ['อวัยวะสำหรับมองเห็น', 'พ่อของพ่อหรือแม่', 'ลวดลายที่มีลักษณะเป็นตาราง'],
    'ยา': ['สารที่ใช้รักษาโรค', 'แม่ของพ่อหรือแม่', 'น้ำยาที่ใช้ในการต่างๆ เช่น ยาสระผม'],
    'เพลา': ['เวลา', 'แกนที่หมุนได้ในเครื่องยนต์', 'ช้าๆ นุ่มนวล'],
    'ลาย': ['เส้นหรือรูปที่ปรากฏบนพื้นผิว', 'ชื่อ ลายเซ็น', 'แตก ร้าว'],
    'ครอง': ['ปกครอง ดูแล', 'สวมใส่ เช่น ครองจีวร', 'ครอบครอง'],
    'เตา': ['อุปกรณ์ให้ความร้อนสำหรับประกอบอาหาร', 'ต่อหู ติ่งหู'],
    'คาง': ['ส่วนที่อยู่ใต้ริมฝีปากล่าง', 'ค้าง ติดค้าง ยังไม่เสร็จ'],
    'ค้าง': ['อยู่ในตำแหน่งเดิมเป็นเวลานาน', 'ยังไม่ชำระเงิน', 'คาคืน ค้างคืน'],
    'นา': ['ที่สำหรับปลูกข้าว', 'คำที่ใช้ท้ายประโยคเพื่อเน้นย้ำหรือชักชวน', 'คำบุรพบท แสดงความเป็นเจ้าของ (ภาษาอีสาน)'],
    'ตี': ['ทุบ ต่อย', 'แปล ตีความ', 'กระทบกัน เช่น ตีกลับ'],
    'กิน': ['รับประทาน', 'ครอบคลุมพื้นที่', 'ได้รับ เช่น กินเงินเดือน'],
    'แล้ว': ['เสร็จสิ้น', 'เหตุการณ์ผ่านไปแล้ว', 'คำเชื่อมบอกเวลาเช่น พอทำแล้วก็ไป'],
    'แค่': ['เพียง', 'ผัก ชนิดหนึ่ง'],
    'ไหว': ['ยกมือขึ้นประนม', 'เคลื่อนไหวเล็กน้อย', 'สามารถทนได้'],
    'เพื่อ': ['สำหรับ', 'เพราะ', 'เพื่อนสนิท (ภาษาถิ่นอีสาน)'],
    'ยัง': ['คงอยู่', 'ทำให้เป็น', 'คำใช้ในประโยคปฏิเสธ'],
    'จ้อง': ['มองตรงไปที่', 'จับจอง ยึดเป็นเจ้าของ'],
    'บรรทัด': ['แถว แนว', 'ใช้เรียกกระดาษที่มีเส้นบรรทัด', 'มาตรฐาน แนวทาง'],
    'ลง': ['เคลื่อนที่ลงสู่ด้านล่าง', 'วางหรือติดตั้ง', 'ติดตาม เข้าไป']
}

# Content domains to generate examples from
DOMAINS = [
    'บทความทั่วไป', 'บทความวิชาการ', 'ข่าว', 'บทสนทนา', 
    'บันเทิง', 'กีฬา', 'ธุรกิจ', 'การศึกษา', 'สุขภาพ', 'เทคโนโลยี'
]

def create_wsd_prompt(word, meanings, domain):
    """
    Create a prompt for generating Thai text with word sense disambiguation examples
    
    Args:
        word (str): The polysemous word
        meanings (list): List of different meanings of the word
        domain (str): Domain of the content
        
    Returns:
        str: The generated prompt
    """
    meanings_text = '\n'.join([f"{i+1}. {meaning}" for i, meaning in enumerate(meanings)])
    
    prompt = f"""สร้างตัวอย่างประโยคภาษาไทยที่ใช้คำว่า "{word}" ในความหมายต่างๆ ในหัวข้อเกี่ยวกับ{domain}

คำว่า "{word}" มีความหมายหลายอย่าง ได้แก่:
{meanings_text}

กรุณาสร้างตัวอย่างประโยคสำหรับแต่ละความหมาย อย่างน้อยความหมายละ 1 ประโยค ที่แสดงการใช้คำที่มีความหมายต่างกันอย่างชัดเจน

กรุณาตอบในรูปแบบนี้:

ความหมายที่ 1: [ความหมายที่ 1]
ตัวอย่างประโยค: [ประโยคที่ใช้คำในความหมายที่ 1]
คำอธิบาย: [อธิบายวิธีสังเกตว่าคำนี้ใช้ในความหมายที่ 1]

ความหมายที่ 2: [ความหมายที่ 2]
ตัวอย่างประโยค: [ประโยคที่ใช้คำในความหมายที่ 2]
คำอธิบาย: [อธิบายวิธีสังเกตว่าคำนี้ใช้ในความหมายที่ 2]

(และทำเช่นนี้สำหรับทุกความหมาย)"""
    
    return prompt

def generate_wsd_samples(api_key, num_samples, domains=None, polysemous_words=None):
    """
    Generate word sense disambiguation samples using Deepseek API
    
    Args:
        api_key (str): Deepseek API key
        num_samples (int): Number of samples to generate
        domains (list): List of domains to generate samples from
        polysemous_words (dict): Dictionary of polysemous words with their meanings
        
    Returns:
        list: Generated samples
    """
    if domains is None:
        domains = DOMAINS
    
    if polysemous_words is None:
        polysemous_words = POLYSEMOUS_WORDS
    
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในภาษาไทยและการแยกแยะความหมายของคำ (Word Sense Disambiguation) คุณเข้าใจคำที่มีหลายความหมายและสามารถสร้างตัวอย่างที่ชัดเจนได้"""
    
    results = []
    
    # Create a list of (word, meanings) pairs from the dictionary
    word_meanings_pairs = [(word, meanings) for word, meanings in polysemous_words.items()]
    
    # If num_samples is more than the number of words, we'll need to repeat some words
    samples_per_word = max(1, num_samples // len(word_meanings_pairs))
    extra_samples = num_samples % len(word_meanings_pairs)
    
    # Create a list of all word-meaning pairs to generate, with repetition as needed
    generation_list = word_meanings_pairs * samples_per_word
    generation_list.extend(random.sample(word_meanings_pairs, extra_samples))
    
    # Shuffle the list to avoid generating the same word consecutively
    random.shuffle(generation_list)
    
    for word, meanings in tqdm(generation_list, desc="Generating WSD samples"):
        domain = random.choice(domains)
        
        prompt = create_wsd_prompt(word, meanings, domain)
        
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
                        # Initialize a list to store examples for each meaning
                        sense_examples = []
                        
                        # Split the response by meaning sections
                        for i, meaning in enumerate(meanings):
                            # Look for section headers like "ความหมายที่ 1:", "ความหมายที่ 2:", etc.
                            meaning_marker = f"ความหมายที่ {i+1}:"
                            example_marker = "ตัวอย่างประโยค:"
                            explanation_marker = "คำอธิบาย:"
                            
                            # If this is the last meaning, the section ends at the end of the response
                            if i < len(meanings) - 1:
                                next_meaning_marker = f"ความหมายที่ {i+2}:"
                                meaning_section = response.split(meaning_marker)[1].split(next_meaning_marker)[0]
                            else:
                                if meaning_marker in response:
                                    meaning_section = response.split(meaning_marker)[1]
                                else:
                                    # If marker not found, skip this meaning
                                    continue
                            
                            # Extract example and explanation from the section
                            meaning_text = meaning
                            example_text = ""
                            explanation_text = ""
                            
                            if example_marker in meaning_section:
                                example_part = meaning_section.split(example_marker)[1]
                                if explanation_marker in example_part:
                                    example_text = example_part.split(explanation_marker)[0].strip()
                                    explanation_text = example_part.split(explanation_marker)[1].strip()
                                else:
                                    example_text = example_part.strip()
                            
                            # Add this sense example to the list
                            if example_text:
                                sense_examples.append({
                                    "sense": meaning_text,
                                    "example": example_text,
                                    "explanation": explanation_text
                                })
                        
                        # Only add the sample if we have at least one valid sense example
                        if sense_examples:
                            sample = {
                                "word": word,
                                "domain": domain,
                                "sense_examples": sense_examples,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            results.append(sample)
                            break
                        else:
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_DELAY)
                            else:
                                print(f"Failed to parse sense examples for '{word}' after {MAX_RETRIES} attempts")
                    except Exception as e:
                        print(f"Error parsing response for '{word}': {e}")
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
    parser = argparse.ArgumentParser(description='Generate Thai Word Sense Disambiguation dataset using Deepseek API')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: output/thai_wsd_dataset_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=20,
                        help='Number of samples to generate')
    parser.add_argument('--domains', type=str, nargs='+', default=None,
                        help='Specific domains to generate (default: all)')
    parser.add_argument('--words', type=str, nargs='+', default=None,
                        help='Specific polysemous words to use (default: all)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Deepseek API key (will use environment variable if not provided)')
    
    args = parser.parse_args()
    
    # Filter polysemous words if specific words are provided
    if args.words:
        polysemous_words = {word: POLYSEMOUS_WORDS[word] for word in args.words if word in POLYSEMOUS_WORDS}
        if not polysemous_words:
            print(f"Error: None of the provided words are in the polysemous words list.")
            print(f"Available words: {', '.join(POLYSEMOUS_WORDS.keys())}")
            return
    else:
        polysemous_words = POLYSEMOUS_WORDS
    
    # Load API key
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: Deepseek API key not found. Please provide it with --api-key or set DEEPSEEK_API_KEY in your .env file.")
        return
    
    # Set default output file if not provided
    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_wsd_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_wsd_samples(api_key, args.samples, args.domains, polysemous_words)
    
    # Save samples
    save_samples(samples, args.output_file)

if __name__ == "__main__":
    main()