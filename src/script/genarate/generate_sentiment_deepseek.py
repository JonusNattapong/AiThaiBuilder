import os
import json
import random
import time
import csv
import argparse
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

# Add project root to Python path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import translation utilities
from utils.deepseek_translation_utils import generate_en_to_th_translation

# Load environment variables from .env file
load_dotenv()

SENTIMENT_TASKS = {
    'basic_emotions': {
        'happy': ['มีความสุข', 'ดีใจ', 'สนุกสนาน', 'ยินดี', 'ตื่นเต้น'],
        'sad': ['เศร้า', 'เสียใจ', 'ผิดหวัง', 'ท้อแท้', 'หดหู่'],
        'angry': ['โกรธ', 'หงุดหงิด', 'ฉุนเฉียว', 'โมโห', 'เดือดดาล']
    },
    'complex_emotions': {
        'love': ['รัก', 'หลงใหล', 'เสน่หา', 'ผูกพัน', 'อบอุ่น'],
        'fear': ['กลัว', 'วิตก', 'หวาดกลัว', 'ตื่นตระหนก', 'หวั่นใจ'],
        'surprise': ['ประหลาดใจ', 'ตกใจ', 'อัศจรรย์ใจ', 'พิศวง', 'งงงวย']
    },
    'social_emotions': {
        'gratitude': ['ขอบคุณ', 'ซาบซึ้ง', 'สำนึกบุญคุณ', 'กตัญญู', 'ประทับใจ'],
        'pride': ['ภูมิใจ', 'มั่นใจ', 'เชิดหน้าชูตา', 'สง่างาม', 'ยโส'],
        'shame': ['อับอาย', 'ละอายใจ', 'เขินอาย', 'หน้าแตก', 'ขายหน้า']
    }
}

TOPICS = [
    'ครอบครัว', 'ความรัก', 'มิตรภาพ', 'การทำงาน', 'การเรียน',
    'กีฬา', 'ดนตรี', 'ภาพยนตร์', 'สุขภาพ', 'การเงิน',
    'สังคม', 'การเมือง', 'สิ่งแวดล้อม', 'การเดินทาง', 'ความฝัน',
    'อาหาร', 'การท่องเที่ยว', 'เทคโนโลยี', 'ศิลปะ', 'วัฒนธรรม'
]

SYSTEM_PROMPTS = {
    'basic_emotions': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่แสดงอารมณ์พื้นฐาน
ให้เน้น:
1. สร้างประโยคภาษาไทยที่เป็นธรรมชาติ สมจริง
2. ใช้คำที่สื่ออารมณ์อย่างชัดเจนตรงไปตรงมา
3. ความยาว 1-2 ประโยค กระชับได้ใจความ
4. เขียนในรูปแบบบทสนทนาหรือความคิดที่พบได้ในชีวิตประจำวัน""",

    'complex_emotions': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่แสดงอารมณ์ซับซ้อน
ให้เน้น:
1. สร้างประโยคที่สื่อความรู้สึกละเอียดอ่อน ลึกซึ้ง
2. ผสมผสานอารมณ์หลายแง่มุมอย่างเป็นธรรมชาติ
3. ใช้ภาษาไทยที่สละสลวย มีชั้นเชิง
4. สร้างเรื่องราวหรือสถานการณ์ที่สมจริง น่าเชื่อถือ""",

    'social_emotions': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่แสดงอารมณ์ทางสังคม
ให้เน้น:
1. ใช้สำนวนภาษาที่สอดคล้องกับวัฒนธรรมไทย
2. สะท้อนค่านิยมและบรรทัดฐานทางสังคมไทย
3. เลือกใช้ระดับภาษาให้เหมาะสมกับสถานการณ์
4. นำเสนอการปฏิสัมพันธ์ระหว่างบุคคลอย่างสมจริง"""
}

# Added example outputs for each emotion
EXAMPLE_OUTPUTS = {
    'happy': [
        'เรียนจบแล้ว! ดีใจจนน้ำตาไหล ความพยายามของเราไม่สูญเปล่า',
        'ได้เจอเพื่อนเก่าโดยบังเอิญ คิดถึงจนกอดกันร้องไห้เลย'
    ],
    'sad': [
        'ดูแลต้นไม้มาตั้งนาน วันนี้มันเหี่ยวตายซะแล้ว ใจหายจัง',
        'เก็บของเก่าแล้วเจอรูปถ่ายครั้งสุดท้ายกับคุณยาย น้ำตาไหลโดยไม่รู้ตัว'
    ],
    'angry': [
        'บอกแล้วว่าอย่าเอาขยะมาทิ้งหน้าบ้าน ยังทำอีก โมโหจริงๆ!',
        'รถติดเพราะคนจอดซ้อนคัน แถมยังเถียงว่าตัวเองถูก หงุดหงิดที่สุด'
    ]
}

TEXT_GENERATION_PARAMS = {
    'temperature': 0.7,
    'max_tokens': 100,
    'top_p': 0.9,
    'frequency_penalty': 0.5,
    'presence_penalty': 0.5
}

MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 20  # Increased for faster generation
BATCH_SIZE = 10   # Process more samples in parallel
OUTPUT_DIR = 'output'

def create_prompt(emotion, topic, examples):
    return f"""สร้างประโยคภาษาไทยที่แสดงอารมณ์ "{emotion}" เกี่ยวกับ "{topic}"

ตัวอย่างประโยคที่แสดงอารมณ์นี้:
1. {examples[0]}
2. {examples[1]}

กรุณาสร้างประโยคใหม่ที่:
1. ใช้ภาษาไทยที่เป็นธรรมชาติ
2. สื่ออารมณ์ได้ชัดเจน
3. เกี่ยวข้องกับหัวข้อที่กำหนด
4. ไม่ซ้ำกับตัวอย่าง

ความยาว: 1-2 ประโยค
ตอบ:"""

def invoke_deepseek_batch(prompts, api_key, max_retries=3, delay=1):
    """Process multiple prompts in parallel"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = []
        for i, (system_prompt, user_prompt) in enumerate(prompts):
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **TEXT_GENERATION_PARAMS
            }
            
            def make_request(prompt_index, retries=0):
                try:
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result and 'choices' in result and result['choices']:
                        text = result['choices'][0]['message']['content'].strip()
                        results[prompt_index] = text
                        return True
                except Exception as e:
                    if retries < max_retries - 1:
                        time.sleep(delay)
                        return make_request(prompt_index, retries + 1)
                    print(f"Request failed after {max_retries} attempts: {str(e)}")
                    return False
            
            futures.append(
                executor.submit(make_request, i)
            )
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()
    
    return results

def generate_samples(api_key, num_samples, progress_bar=None):
    """Generate samples with improved batching and error handling"""
    results = defaultdict(list)
    translation_cache = {}
    
    # First, ensure all system prompts are translated
    print("Translating system prompts...")
    translated_prompts = {}
    for category, prompt in SYSTEM_PROMPTS.items():
        if prompt not in translation_cache:
            translation = generate_en_to_th_translation(prompt, api_key, "Use formal Thai language")
            translation_cache[prompt] = translation
        translated_prompts[category] = translation_cache[prompt]
    
    all_prompts = []
    for category, emotions in SENTIMENT_TASKS.items():
        for emotion, _ in emotions.items():
            examples = EXAMPLE_OUTPUTS.get(emotion, ["", ""])
            for _ in range(num_samples):
                topic = random.choice(TOPICS)
                prompt = create_prompt(emotion, topic, examples)
                all_prompts.append((translated_prompts[category], prompt, category, emotion))
    
    # Process prompts in larger batches
    for i in range(0, len(all_prompts), BATCH_SIZE):
        batch = all_prompts[i:i + BATCH_SIZE]
        prompts = [(p[0], p[1]) for p in batch]
        
        # Try the batch with retries
        for attempt in range(MAX_RETRIES):
            try:
                batch_results = invoke_deepseek_batch(prompts, api_key)
                
                # Store successful results
                for (_, _, category, emotion), text in zip(batch, batch_results):
                    if text:
                        results[f"{category}_{emotion}"].append({
                            'text': text,
                            'label': emotion
                        })
                
                if progress_bar:
                    progress_bar.update(len(batch))
                    
                break  # Success, move to next batch
                
            except Exception as e:
                print(f"Batch attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate Thai sentiment dataset')
    parser.add_argument('--samples', type=int, default=50,
                      help='Number of samples per emotion')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return
    
    # Calculate total samples
    total_emotions = sum(len(emotions) for emotions in SENTIMENT_TASKS.values())
    total_samples = total_emotions * args.samples
    
    print(f"\nGenerating Thai sentiment dataset:")
    print(f"- {args.samples} samples per emotion")
    print(f"- {total_emotions} emotions")
    print(f"- {total_samples} total samples")
    print("\nStarting generation...\n")
    
    # Generate samples with progress bar
    with tqdm(total=total_samples, desc="Generating samples") as pbar:
        results = generate_samples(api_key, args.samples, pbar)
    
    # Save results by category
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for category_emotion, samples in results.items():
        if samples:
            category = category_emotion.split('_')[0]
            # Save as JSON
            json_file = os.path.join(OUTPUT_DIR, f"thai_sentiment_{category}_{timestamp}.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            
            # Save as Parquet
            df = pd.DataFrame(samples)
            parquet_file = os.path.join(OUTPUT_DIR, f"thai_sentiment_{category}_{timestamp}.parquet")
            df.to_parquet(parquet_file, engine='pyarrow', index=False)
            
            print(f"Saved {len(samples)} samples to:")
            print(f"- JSON: {json_file}")
            print(f"- Parquet: {parquet_file}")
    
    # Save combined results
    all_samples = [item for sublist in results.values() for item in sublist]
    
    # Save combined JSON
    json_output = os.path.join(OUTPUT_DIR, f"thai_sentiment_dataset_{len(all_samples)}_{timestamp}.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    # Save combined Parquet
    df_all = pd.DataFrame(all_samples)
    parquet_output = os.path.join(OUTPUT_DIR, f"thai_sentiment_dataset_{len(all_samples)}_{timestamp}.parquet")
    df_all.to_parquet(parquet_output, engine='pyarrow', index=False)
    
    print(f"\nGeneration complete!")
    print(f"Total samples generated: {len(all_samples)}")
    print(f"Combined dataset saved to:")
    print(f"- JSON: {json_output}")
    print(f"- Parquet: {parquet_output}")

if __name__ == "__main__":
    main()