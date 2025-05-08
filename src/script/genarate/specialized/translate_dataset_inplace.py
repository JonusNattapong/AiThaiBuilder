#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset In-Place Translation Script

This script translates datasets between languages (en-th, zh-th, hi-th) and overwrites the original files
to maintain dataset integrity. It supports various dataset formats including JSON, CSV, and JSONL.
Fields to translate can be specified or automatically detected.

Usage:
    python translate_dataset_inplace.py --input [INPUT_FILE] --source-lang [SRC_LANG] --target-lang [TGT_LANG] [--fields FIELD1 FIELD2 ...]
    
Example:
    python translate_dataset_inplace.py --input dataset.json --source-lang hi --target-lang th --fields text content
    python translate_dataset_inplace.py --input dataset.csv --source-lang hi --target-lang th  # Auto-detect fields
"""

import os
import sys
import json
import csv
import argparse
import logging
import time
import datetime
import threading
import queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional
from retrying import retry
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Download data for METEOR
nltk.download('wordnet', quiet=True)

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utils
from utils.deepseek_utils import get_deepseek_api_key, generate_with_deepseek
from utils.deepseek_translation_utils import (
    generate_translation_sample,
    detect_language_with_deepseek,
    generate_reference_translation,
    check_translation_quality,
    refine_translation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', f'translation_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), 'w', 'utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

# Constants
SUPPORTED_SOURCE_LANGS = ['en', 'zh', 'hi']  # เพิ่ม hi
SUPPORTED_TARGET_LANGS = ['th']
SUPPORTED_FILE_FORMATS = ['.json', '.jsonl', '.csv']
DEFAULT_BATCH_SIZE = 5
MAX_THREADS = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONTENT_MAX_LENGTH = 10000  # Maximum length of content to translate in a single API call
QUALITY_CHECK_PROBABILITY = 0.1
QUALITY_THRESHOLD = 0.5
METRIC_WEIGHTS = {'bleu': 0.3, 'rouge': 0.3, 'meteor': 0.4}

# Translation progress tracking
class TranslationProgress:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.refinements = 0
        self.lock = threading.Lock()
        
    def update(self, success=True, refined=False):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            if refined:
                self.refinements += 1
                
    def get_stats(self):
        with self.lock:
            return {
                'total': self.total,
                'completed': self.completed,
                'successful': self.successful,
                'failed': self.failed,
                'refinements': self.refinements,
                'progress_percent': (self.completed / self.total * 100) if self.total > 0 else 0
            }

def detect_file_format(file_path):
    """Detect file format based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FILE_FORMATS:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {', '.join(SUPPORTED_FILE_FORMATS)}")
    return ext

def load_dataset(file_path):
    """Load dataset from file."""
    file_format = detect_file_format(file_path)
    
    try:
        if file_format == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_format == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        elif file_format == '.csv':
            data = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            return data
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
        
    return None

def save_dataset(data, file_path, original_format=None):
    """Save dataset to file, preserving original format."""
    if original_format is None:
        original_format = detect_file_format(file_path)
    
    # Create backup of original file
    backup_path = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Created backup of original dataset at: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup file: {str(e)}")
        raise ValueError("Aborting to prevent data loss. Please ensure the file is accessible.")
    
    # Save translated dataset
    try:
        if original_format == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif original_format == '.jsonl':
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif original_format == '.csv':
            if data:
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        logger.info(f"Successfully saved translated dataset to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving translated dataset: {str(e)}")
        logger.info(f"Original dataset backup is available at: {backup_path}")
        return False

def find_fields_to_translate(data: List[Dict], source_lang: str, api_key: str, sample_size: int = 5) -> List[str]:
    """
    Automatically detect fields containing text in the source language using Deepseek.
    """
    field_counts = {}
    text_samples = {}

    # Collect samples from all fields
    for item in data[:100]:  # Limit to first 100 items to avoid excessive API calls
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, str) and value.strip() and value.strip() != "nan":
                if key not in text_samples:
                    text_samples[key] = []
                text_samples[key].append(value)
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                if key not in text_samples:
                    text_samples[key] = []
                text_samples[key].extend([v for v in value if v.strip() and v.strip() != "nan"])

    # Detect language for sampled texts
    fields_to_translate = []
    for field, samples in text_samples.items():
        if not samples:
            continue
        # Sample up to sample_size texts
        sample_texts = random.sample(samples, min(sample_size, len(samples)))
        source_lang_count = 0

        for text in sample_texts:
            detected_lang = detect_language_with_deepseek(text, api_key)
            if detected_lang == source_lang.upper():
                source_lang_count += 1

        # Select field if more than 50% of samples are in source language
        if source_lang_count / len(sample_texts) > 0.5:
            fields_to_translate.append(field)
            logger.info(f"Selected field '{field}' for translation (source language detected in {source_lang_count}/{len(sample_texts)} samples)")

    if not fields_to_translate:
        logger.warning("No fields found with source language texts")
    return fields_to_translate

def translate_text(text, source_lang, target_lang, api_key, additional_instructions="", max_retries=DEFAULT_MAX_RETRIES):
    """Translate text from source language to target language with quality checking."""
    if not text or len(text.strip()) == 0 or text.strip() == "nan":
        return text, False
    
    # Handle excessively long text by splitting into chunks
    if len(text) > DEFAULT_CONTENT_MAX_LENGTH:
        logger.warning(f"Text exceeds maximum length ({len(text)} > {DEFAULT_CONTENT_MAX_LENGTH}). Splitting for translation.")
        chunks = []
        for i in range(0, len(text), DEFAULT_CONTENT_MAX_LENGTH):
            chunks.append(text[i:i+DEFAULT_CONTENT_MAX_LENGTH])
        
        translated_chunks = []
        was_refined = False
        for chunk in chunks:
            translated_chunk, chunk_refined = translate_text(chunk, source_lang, target_lang, api_key, additional_instructions, max_retries)
            translated_chunks.append(translated_chunk)
            if chunk_refined:
                was_refined = True
        
        return "".join(translated_chunks), was_refined
    
    # Attempt translation with retries
    for attempt in range(max_retries):
        try:
            translated = generate_translation_sample(text, source_lang, target_lang, api_key, additional_instructions)
            if translated and not translated.startswith("Error"):
                # Quality check
                final_translation = translated
                was_refined = False
                if random.random() < QUALITY_CHECK_PROBABILITY:
                    reference_text = generate_reference_translation(text, source_lang, target_lang, api_key)
                    if reference_text:
                        quality_score = check_translation_quality(translated, reference_text, api_key)
                        logger.info(f"Quality scores for text: BLEU={quality_score['bleu']:.2f}, ROUGE={quality_score['rouge']:.2f}, METEOR={quality_score['meteor']:.2f}, Average={quality_score['average']:.2f}")
                        
                        if quality_score['average'] < QUALITY_THRESHOLD:
                            refined_text = refine_translation(text, translated, source_lang, target_lang, api_key, quality_score)
                            refined_score = check_translation_quality(refined_text, reference_text, api_key)
                            if refined_score['average'] > quality_score['average']:
                                final_translation = refined_text
                                was_refined = True
                                logger.info(f"Refined translation, new average score: {refined_score['average']:.2f}")
                return final_translation, was_refined
            
            logger.warning(f"Translation attempt {attempt+1}/{max_retries} failed: {translated[:100]}...")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Translation error (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    # If all retries fail, return original text
    logger.error(f"Translation failed after {max_retries} attempts. Keeping original text.")
    return text, False

def process_nested_dict(obj, fields_to_translate, source_lang, target_lang, api_key, translation_queue, path=""):
    """Process nested dictionaries and translate specified fields."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            if key in fields_to_translate:
                if isinstance(value, str):
                    translation_queue.put((value, current_path, obj, key))
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    for i, item in enumerate(value):
                        translation_queue.put((item, f"{current_path}[{i}]", value, i))
            
            if isinstance(value, (dict, list)):
                process_nested_dict(value, fields_to_translate, source_lang, target_lang, api_key, translation_queue, current_path)
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{path}[{i}]"
            process_nested_dict(item, fields_to_translate, source_lang, target_lang, api_key, translation_queue, current_path)

def translation_worker(queue, source_lang, target_lang, api_key, progress, max_retries=DEFAULT_MAX_RETRIES):
    """Worker function to process translations from the queue."""
    while True:
        try:
            item = queue.get(block=False)
            if item is None:
                break
                
            text, path, parent, key = item
            
            try:
                translated_text, was_refined = translate_text(text, source_lang, target_lang, api_key, max_retries=max_retries)
                parent[key] = translated_text
                progress.update(success=True, refined=was_refined)
            except Exception as e:
                logger.error(f"Error translating text at {path}: {str(e)}")
                progress.update(success=False)
            
            time.sleep(0.5)
            
        except queue.Empty:
            break

def translate_dataset(data, fields_to_translate, source_lang, target_lang, api_key, num_threads=MAX_THREADS):
    """Translate dataset fields."""
    translation_queue = queue.Queue()
    
    process_nested_dict(data, fields_to_translate, source_lang, target_lang, api_key, translation_queue)
    
    queue_size = translation_queue.qsize()
    logger.info(f"Found {queue_size} items to translate")
    
    if queue_size == 0:
        logger.warning(f"No fields matching {fields_to_translate} found in the dataset for translation")
        return data
    
    progress = TranslationProgress(queue_size)
    
    queues = []
    items = []
    while not translation_queue.empty():
        try:
            items.append(translation_queue.get(block=False))
        except queue.Empty:
            break
    
    num_workers = min(num_threads, len(items))
    if num_workers == 0:
        return data
        
    items_per_worker = len(items) // num_workers
    remainder = len(items) % num_workers
    
    start_idx = 0
    for i in range(num_workers):
        worker_queue = queue.Queue()
        worker_items = items_per_worker + (1 if i < remainder else 0)
        
        for j in range(start_idx, start_idx + worker_items):
            worker_queue.put(items[j])
        
        start_idx += worker_items
        queues.append(worker_queue)
    
    threads = []
    for i in range(num_workers):
        thread = threading.Thread(
            target=translation_worker,
            args=(queues[i], source_lang, target_lang, api_key, progress)
        )
        thread.start()
        threads.append(thread)
    
    with tqdm(total=queue_size, desc=f"Translating {source_lang} to {target_lang}") as pbar:
        completed_prev = 0
        while any(thread.is_alive() for thread in threads):
            stats = progress.get_stats()
            completed_current = stats['completed']
            pbar.update(completed_current - completed_prev)
            completed_prev = completed_current
            time.sleep(0.1)
    
    for thread in threads:
        thread.join()
    
    stats = progress.get_stats()
    logger.info(f"Translation completed: {stats['successful']} successful, {stats['failed']} failed, {stats['refinements']} refined")
    
    return data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Translate dataset fields and overwrite the original file')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input dataset file path')
    
    parser.add_argument('--source-lang', type=str, required=True, choices=SUPPORTED_SOURCE_LANGS,
                        help='Source language code')
    
    parser.add_argument('--target-lang', type=str, required=True, choices=SUPPORTED_TARGET_LANGS,
                        help='Target language code')
    
    parser.add_argument('--fields', type=str, nargs='*',
                        help='Field names to translate (leave empty for auto-detection)')
    
    parser.add_argument('--api-key', type=str,
                        help='Deepseek API key (will use environment variable if not provided)')
    
    parser.add_argument('--threads', type=int, default=MAX_THREADS,
                        help=f'Number of translation threads (default: {MAX_THREADS})')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run without saving changes')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    api_key = args.api_key or get_deepseek_api_key()
    if not api_key:
        logger.error("Deepseek API key not found. Please provide it as an argument or set it in your environment.")
        sys.exit(1)
    
    try:
        logger.info(f"Loading dataset from {args.input}")
        data = load_dataset(args.input)
        file_format = detect_file_format(args.input)
        logger.info(f"Dataset loaded: {file_format} format")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Determine fields to translate
    fields_to_translate = args.fields if args.fields else []
    if not fields_to_translate:
        logger.info(f"No fields specified. Auto-detecting fields with {args.source_lang} text.")
        fields_to_translate = find_fields_to_translate(data, args.source_lang, api_key)
        if not fields_to_translate:
            logger.error("No fields found to translate. Exiting.")
            sys.exit(1)
    else:
        # Validate specified fields
        sample_item = data[0] if data and isinstance(data, list) and isinstance(data[0], dict) else {}
        missing_fields = [f for f in fields_to_translate if f not in sample_item]
        if missing_fields:
            logger.error(f"Specified fields not found in dataset: {missing_fields}")
            sys.exit(1)
        logger.info(f"Using specified fields for translation: {fields_to_translate}")
    
    try:
        logger.info(f"Starting translation from {args.source_lang} to {args.target_lang} for fields: {', '.join(fields_to_translate)}")
        translated_data = translate_dataset(
            data=data,
            fields_to_translate=fields_to_translate,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            api_key=api_key,
            num_threads=args.threads
        )
        logger.info("Translation completed")
    except Exception as e:
        logger.error(f"Translation process failed: {str(e)}")
        sys.exit(1)
    
    if not args.dry_run:
        try:
            logger.info(f"Saving translated dataset to {args.input} (original file will be backed up)")
            success = save_dataset(translated_data, args.input, file_format)
            if success:
                logger.info("Dataset successfully translated and saved")
            else:
                logger.error("Failed to save translated dataset")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            sys.exit(1)
    else:
        logger.info("Dry run completed - no changes saved")
        
    logger.info("Translation process complete")

if __name__ == "__main__":
    main()