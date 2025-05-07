import os
import json
import random
import time
import csv
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config_generate_sentiment import (
    SENTIMENT_TASKS,
    TOPICS,
    EMOTION_SPECIFIC_TOPICS,
    SYSTEM_PROMPTS,
    SCENE_TEMPLATES,
    TEXT_GENERATION_PARAMS,
    MAX_RETRIES,
    RETRY_DELAY,
    NUM_SAMPLES_PER_TASK,
    OUTPUT_DIR
)

def invoke_deepseek(system_prompt, user_prompt, api_key):
    """Call DeepSeek API with retry mechanism"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **TEXT_GENERATION_PARAMS
    }

    for attempt in range(MAX_RETRIES):
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
                return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                
    return None

def generate_sentiment_text(category, emotion, api_key):
    """Generate text for a given emotion category and emotion"""
    # Get emotion words and appropriate topics
    emotion_words = SENTIMENT_TASKS[category][emotion]
    topics = EMOTION_SPECIFIC_TOPICS.get(emotion, TOPICS)
    topic = random.choice(topics)
    
    # Get appropriate system prompt and template
    system_prompt = SYSTEM_PROMPTS[category]
    template = random.choice(SCENE_TEMPLATES.get(emotion, SCENE_TEMPLATES['happy']))  # Default to happy template if none exists
    selected_words = random.sample(emotion_words, min(3, len(emotion_words)))
    
    user_prompt = (
        f"สร้างข้อความภาษาไทยที่แสดงความรู้สึก {emotion} เกี่ยวกับเรื่อง {topic} "
        f"โดยใช้คำที่แสดงอารมณ์เช่น: {', '.join(selected_words)} "
        f"ให้เขียนตามรูปแบบนี้: {template} "
        "ให้ข้อความมีความยาวประมาณ 1-2 ประโยค สมจริง และเป็นธรรมชาติ"
    )
    
    response = invoke_deepseek(system_prompt, user_prompt, api_key)
    return response

def generate_dataset(output_file, api_key, max_workers=4):
    """Generate sentiment dataset for all emotion categories"""
    samples = []
    
    def generate_sample(category, emotion):
        text = generate_sentiment_text(category, emotion, api_key)
        if text:
            return {
                'text': text,
                'emotion': emotion,
                'category': category,
                'intensity': random.uniform(0.7, 1.0),  # High intensity for training
                'confidence': 0.9  # High confidence threshold
            }
        return None

    # Generate samples for each category and emotion
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for category, emotions in SENTIMENT_TASKS.items():
            for emotion in emotions:
                for _ in range(NUM_SAMPLES_PER_TASK):
                    futures.append(executor.submit(generate_sample, category, emotion))
        
        with tqdm(total=len(futures), desc="Generating samples") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    samples.append(result)
                pbar.update(1)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'emotion', 'category', 'intensity', 'confidence'])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"\nGenerated {len(samples)} total samples:")
    for category, emotions in SENTIMENT_TASKS.items():
        print(f"\n{category}:")
        for emotion in emotions:
            count = sum(1 for s in samples if s['emotion'] == emotion)
            print(f"  {emotion}: {count} samples")
    print(f"\nSaved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive sentiment dataset')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                      help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    output_file = os.path.join(args.output_dir, 'thai_sentiment_comprehensive.csv')
    generate_dataset(output_file, api_key, args.max_workers)

if __name__ == '__main__':
    main()