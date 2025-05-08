#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset In-Place Translation Script

This script translates datasets between languages (en-th, zh-th) and overwrites the original files
to maintain dataset integrity. It supports various dataset formats including JSON, CSV, and JSONL.

Usage:
    python translate_dataset_inplace.py --input [INPUT_FILE] --source-lang [SRC_LANG] --target-lang [TGT_LANG] [--fields FIELD1 FIELD2 ...]
    
Example:
    python translate_dataset_inplace.py --input dataset.json --source-lang en --target-lang th --fields text content
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

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utils
from utils.deepseek_utils import get_deepseek_api_key, generate_with_deepseek
from utils.deepseek_translation_utils import (
    generate_en_to_th_translation,
    generate_zh_to_th_translation
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
SUPPORTED_SOURCE_LANGS = ['en', 'zh']
SUPPORTED_TARGET_LANGS = ['th']
SUPPORTED_FILE_FORMATS = ['.json', '.jsonl', '.csv']
DEFAULT_BATCH_SIZE = 5
MAX_THREADS = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONTENT_MAX_LENGTH = 10000  # Maximum length of content to translate in a single API call

# Translation progress tracking
class TranslationProgress:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
        
    def update(self, success=True):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
                
    def get_stats(self):
        with self.lock:
            return {
                'total': self.total,
                'completed': self.completed,
                'successful': self.successful,
                'failed': self.failed,
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

def translate_text(text, source_lang, target_lang, api_key, additional_instructions="", max_retries=DEFAULT_MAX_RETRIES):
    """Translate text from source language to target language."""
    if not text or len(text.strip()) == 0:
        return text
    
    # Handle excessively long text by splitting into chunks if needed
    if len(text) > DEFAULT_CONTENT_MAX_LENGTH:
        logger.warning(f"Text exceeds maximum length ({len(text)} > {DEFAULT_CONTENT_MAX_LENGTH}). Splitting for translation.")
        # Simple split approach - could be improved to handle sentences better
        chunks = []
        for i in range(0, len(text), DEFAULT_CONTENT_MAX_LENGTH):
            chunks.append(text[i:i+DEFAULT_CONTENT_MAX_LENGTH])
        
        translated_chunks = []
        for chunk in chunks:
            translated_chunk = translate_text(chunk, source_lang, target_lang, api_key, additional_instructions, max_retries)
            translated_chunks.append(translated_chunk)
        
        return "".join(translated_chunks)
    
    # Choose appropriate translation function
    if source_lang == 'en' and target_lang == 'th':
        translate_func = generate_en_to_th_translation
    elif source_lang == 'zh' and target_lang == 'th':
        translate_func = generate_zh_to_th_translation
    else:
        # Fallback for unsupported language pairs - construct a generic translation request
        def generic_translate(text, api_key, additional_instructions=""):
            system_prompt = f"You are an expert translator from {source_lang} to {target_lang}."
            user_prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"
            if additional_instructions:
                user_prompt += f"\n\nAdditional instructions: {additional_instructions}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return generate_with_deepseek(messages, api_key)
        
        translate_func = generic_translate
    
    # Attempt translation with retries
    for attempt in range(max_retries):
        try:
            translated = translate_func(text, api_key, additional_instructions)
            if translated and not translated.startswith("Error"):
                return translated
            
            logger.warning(f"Translation attempt {attempt+1}/{max_retries} failed: {translated[:100]}...")
            time.sleep(2)  # Short delay between retries
            
        except Exception as e:
            logger.error(f"Translation error (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    # If all retries fail, return original text
    logger.error(f"Translation failed after {max_retries} attempts. Keeping original text.")
    return text

def process_nested_dict(obj, fields_to_translate, source_lang, target_lang, api_key, translation_queue, path=""):
    """Process nested dictionaries and translate specified fields."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            if key in fields_to_translate:
                if isinstance(value, str):
                    # Add to translation queue rather than translating immediately
                    translation_queue.put((value, current_path, obj, key))
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    # For lists of strings, queue each item for translation
                    for i, item in enumerate(value):
                        translation_queue.put((item, f"{current_path}[{i}]", value, i))
            
            # Continue recursion for nested objects
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
            if item is None:  # Sentinel value
                break
                
            text, path, parent, key = item
            
            try:
                translated_text = translate_text(text, source_lang, target_lang, api_key, max_retries=max_retries)
                # Update the original data structure
                parent[key] = translated_text
                progress.update(success=True)
            except Exception as e:
                logger.error(f"Error translating text at {path}: {str(e)}")
                progress.update(success=False)
            
            # Short delay to avoid API rate limits
            time.sleep(0.5)
            
        except queue.Empty:
            break

def translate_dataset(data, fields_to_translate, source_lang, target_lang, api_key, num_threads=MAX_THREADS):
    """Translate dataset fields."""
    # Create a queue for translation tasks
    translation_queue = queue.Queue()
    
    # First pass: identify all fields that need translation and add to queue
    process_nested_dict(data, fields_to_translate, source_lang, target_lang, api_key, translation_queue)
    
    # Get queue size for progress tracking
    queue_size = translation_queue.qsize()
    logger.info(f"Found {queue_size} items to translate")
    
    if queue_size == 0:
        logger.warning(f"No fields matching {fields_to_translate} found in the dataset for translation")
        return data
    
    # Initialize progress tracker
    progress = TranslationProgress(queue_size)
    
    # Create a copy of the queue for each thread
    queues = []
    items = []
    while not translation_queue.empty():
        try:
            items.append(translation_queue.get(block=False))
        except queue.Empty:
            break
    
    # Create balanced queues for workers
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
    
    # Create and start worker threads
    threads = []
    for i in range(num_workers):
        thread = threading.Thread(
            target=translation_worker,
            args=(queues[i], source_lang, target_lang, api_key, progress)
        )
        thread.start()
        threads.append(thread)
    
    # Show progress
    with tqdm(total=queue_size, desc=f"Translating {source_lang} to {target_lang}") as pbar:
        completed_prev = 0
        while any(thread.is_alive() for thread in threads):
            stats = progress.get_stats()
            completed_current = stats['completed']
            pbar.update(completed_current - completed_prev)
            completed_prev = completed_current
            time.sleep(0.1)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Final statistics
    stats = progress.get_stats()
    logger.info(f"Translation completed: {stats['successful']} successful, {stats['failed']} failed")
    
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
    
    parser.add_argument('--fields', type=str, nargs='+', required=True,
                        help='Field names to translate')
    
    parser.add_argument('--api-key', type=str,
                        help='Deepseek API key (will use environment variable if not provided)')
    
    parser.add_argument('--threads', type=int, default=MAX_THREADS,
                        help=f'Number of translation threads (default: {MAX_THREADS})')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run without saving changes')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or get_deepseek_api_key()
    if not api_key:
        logger.error("Deepseek API key not found. Please provide it as an argument or set it in your environment.")
        sys.exit(1)
    
    # Load dataset
    try:
        logger.info(f"Loading dataset from {args.input}")
        data = load_dataset(args.input)
        file_format = detect_file_format(args.input)
        logger.info(f"Dataset loaded: {file_format} format")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Translate dataset
    try:
        logger.info(f"Starting translation from {args.source_lang} to {args.target_lang} for fields: {', '.join(args.fields)}")
        translated_data = translate_dataset(
            data=data,
            fields_to_translate=args.fields,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            api_key=api_key,
            num_threads=args.threads
        )
        logger.info("Translation completed")
    except Exception as e:
        logger.error(f"Translation process failed: {str(e)}")
        sys.exit(1)
    
    # Save translated dataset (if not dry run)
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