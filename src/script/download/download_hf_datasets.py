# -*- coding: utf-8 -*-
"""
Script to download datasets from Hugging Face Hub and save them locally.

Example Usage:
python src/script/download/download_hf_datasets.py --datasets USERNAME/DATASET_NAME1 USERNAME/DATASET_NAME2
python src/script/download/download_hf_datasets.py -d USERNAME/DATASET_NAME

Requires:
- datasets library: pip install datasets
- python-dotenv: pip install python-dotenv
"""

import os
import logging
import argparse
from datasets import load_dataset, DownloadConfig, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from huggingface_hub import HfFolder # To potentially save token
import requests # Added import
import json # Added import
from dotenv import load_dotenv # Add this import

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
# Determine the base path (assuming this script is in the 'Script' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the target base directory to save datasets
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'DatasetDownload')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub.")
    parser.add_argument(
        "-d", "--datasets",
        nargs='+',  # Allows specifying one or more dataset IDs
        required=True,
        help="List of dataset IDs from Hugging Face Hub (e.g., 'username/dataset_name' or 'dataset_name')."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"The base directory where datasets will be saved. Defaults to: {DEFAULT_DOWNLOAD_DIR}"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face Hub token for accessing private datasets. If not provided, uses cached login or public access."
    )
    parser.add_argument(
        "--save_token",
        action="store_true",
        help="If provided with --token, save the token for future use."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional: Specify a specific subset/configuration of the dataset to download."
    )
    return parser.parse_args()

# --- DeepSeek Translation Function ---
def dict_update(d: dict, update_dict: dict) -> None:
    """Safely update a dictionary with another dictionary."""
    for k, v in update_dict.items():
        d[k] = v

def api_deepseek_medical_translation(text_to_translate, source_lang="en", target_lang="th"):
    """Translates text using DeepSeek API, tailored for medical content."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/deepseek-llm-7b-chat")
    # Standard DeepSeek API URL; consider making this configurable if needed
    deepseek_api_url = "https://api.deepseek.com/chat/completions"

    if not api_key:
        logging.error("DEEPSEEK_API_KEY not found in environment variables. Cannot translate.")
        return None

    # Enhanced system prompt based on source language
    if source_lang == "zh":
        system_prompt = "您是一位专业的中泰医学翻译专家。请准确翻译以下文本，保持医学术语的准确性和临床正确性。"
    else:  # default to English-Thai translation expert
        system_prompt = "You are an expert medical translator. Translate the following text accurately, preserving medical terminology and ensuring clinical correctness."

    # Enhanced user prompt with explicit language reference
    if source_lang == "zh":
        user_prompt = f"请将以下中文医学文本翻译成泰语:\n\n原文:\n{text_to_translate}\n\n泰语翻译:"
    else:
        user_prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text_to_translate}"

    payload = {
        "model": deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.5, # Adjusted for potentially more factual translation
        "max_tokens": 1024  # Increased for potentially longer medical texts
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(deepseek_api_url, headers=headers, json=payload, timeout=120) # Increased timeout
        response.raise_for_status()
        result = response.json()
        if result.get("choices") and result["choices"][0].get("message") and result["choices"][0]["message"].get("content"):
            return result["choices"][0]["message"]["content"].strip()
        else:
            logging.error(f"DeepSeek API response format error: {result}")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"DeepSeek API request timed out for text: {text_to_translate[:100]}...")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling DeepSeek API: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during DeepSeek API call: {e}")
        return None

# --- Download Logic ---
def download_and_save_datasets(dataset_ids, base_download_dir, token=None, subset=None):
    """
    Downloads datasets from Hugging Face Hub and saves them to disk.

    Args:
        dataset_ids (list): A list of dataset IDs to download.
        base_download_dir (str): The base directory to save datasets.
        token (str, optional): Hugging Face Hub token. Defaults to None.
        subset (str, optional): Specific dataset subset/configuration. Defaults to None.
    """
    logging.info(f"Base download directory set to: {base_download_dir}")
    if not os.path.exists(base_download_dir):
        try:
            os.makedirs(base_download_dir)
            logging.info(f"Created base directory: {base_download_dir}")
        except OSError as e:
            logging.error(f"Error creating base directory {base_download_dir}: {e}")
            return # Stop if base directory cannot be created

    from typing import Dict, Any, List, Union
    import copy

    # Configure download settings, including the token if provided
    download_config = DownloadConfig(token=token)

    # Translation configuration
    translation_config = {
        'datasets': {
            "FreedomIntelligence/medical-o1-reasoning-SFT": {"source_lang": "en"},
            "hfl/chinese-medical-dialogue": {"source_lang": "zh"}
        },
        'text_column': "text",
        'target_lang': "th"
    }

    for dataset_id in dataset_ids:
        logging.info(f"--- Processing dataset: {dataset_id} ---")
        # Create a safe directory name from the dataset ID
        safe_subdir_name = dataset_id.replace('/', '_') # Replace slashes for directory compatibility
        target_dir = os.path.join(base_download_dir, safe_subdir_name)

        logging.info(f"Target save directory: {target_dir}")

        if os.path.exists(target_dir):
            logging.warning(f"Directory already exists: {target_dir}. Skipping download for {dataset_id}. Delete the directory to re-download.")
            continue

        try:
            # Load the dataset (this handles the download)
            logging.info(f"Attempting to download '{dataset_id}'{f' (subset: {subset})' if subset else ''}...")
            # Try to load dataset in streaming mode first to check available splits
            streaming_dataset = load_dataset(
                dataset_id,
                name=subset,  # Pass subset name if provided
                download_config=download_config,
                streaming=True
            )
            
            # Then load the regular dataset for normal processing
            dataset = load_dataset(
                dataset_id,
                name=subset,
                download_config=download_config
            )
            logging.info(f"Successfully loaded dataset '{dataset_id}'.")

            # Convert and save the dataset to the target directory
            logging.info(f"Saving dataset to {target_dir}...")
            if isinstance(dataset, (Dataset, DatasetDict)):
                dataset.save_to_disk(target_dir)
            elif isinstance(dataset, (IterableDataset, IterableDatasetDict)):
                try:
                    # Handle both single IterableDataset and IterableDatasetDict
                    converted = DatasetDict()
                    
                    # For a single IterableDataset, treat it as having a single 'train' split
                    if isinstance(dataset, IterableDataset):
                        splits_to_process = ['train']
                    elif isinstance(streaming_dataset, IterableDatasetDict):
                        splits_to_process = list(streaming_dataset.keys())
                    else:
                        logging.error(f"Unexpected dataset type: {type(dataset)}")
                        raise ValueError(f"Unexpected dataset type: {type(dataset)}")
                    
                    for split_name in splits_to_process:
                        logging.info(f"Converting split: {split_name}")
                        # Get examples from the streaming dataset with proper type checking
                        examples = []
                        split_data = streaming_dataset[split_name] if isinstance(streaming_dataset, IterableDatasetDict) else streaming_dataset
                        for example in split_data:
                            examples.append(example)
                        converted[split_name] = Dataset.from_list(examples)
                    
                    converted.save_to_disk(target_dir)
                    logging.info(f"Successfully converted and saved iterable dataset")
                except Exception as conversion_error:
                    logging.error(f"Error converting iterable dataset: {conversion_error}")
                    raise
            logging.info(f"Successfully saved '{dataset_id}' to {target_dir}")

            # --- Translation Configuration already defined above ---

            if dataset_id in translation_config['datasets']:
                source_lang = translation_config['datasets'][dataset_id]['source_lang']
                logging.info(f"Translation configuration: {source_lang} -> {translation_config['target_lang']}")
                logging.info(f"Starting medical translation for dataset: {dataset_id}")
                
                # 'dataset' here is the one loaded by load_dataset (non-streaming version)
                actual_dataset_to_process = dataset 
                translated_output_dir = os.path.join(target_dir, "translated_content")
                if not os.path.exists(translated_output_dir):
                    os.makedirs(translated_output_dir)
                    logging.info(f"Created directory for translated content: {translated_output_dir}")

                if isinstance(actual_dataset_to_process, DatasetDict):
                    translated_splits = {}
                    for split_name, split_data in actual_dataset_to_process.items():
                        logging.info(f"Translating split: {split_name} for dataset {dataset_id}")
                        translated_examples = []
                        text_column = translation_config['text_column']
                        target_lang = translation_config['target_lang']
                        
                        for i, example in enumerate(split_data):
                            # Convert example to dictionary if it's not already
                            example_dict = dict(example)
                            new_example = dict(example_dict)
                            
                            # Get text using dictionary access
                            original_text = example_dict.get(text_column)
                            
                            if original_text and isinstance(original_text, str):
                                translated_text = api_deepseek_medical_translation(
                                    text_to_translate=original_text,
                                    source_lang=source_lang,
                                    target_lang=target_lang
                                )
                                # Create translation key
                                translation_key = f"translated_{text_column}_{target_lang}"
                                dict_update(new_example, {
                                    translation_key: translated_text if translated_text else "TRANSLATION_FAILED"
                                })
                            elif not original_text:
                                logging.debug(f"Row {i+1} in split {split_name}: Column '{text_column}' not found or empty.")
                            elif not isinstance(original_text, str):
                                logging.warning(f"Row {i+1} in split {split_name}: Column '{text_column}' is not a string (type: {type(original_text)}). Skipping translation.")
                            translated_examples.append(new_example)
                            if (i + 1) % 100 == 0:
                                logging.info(f"Translated {i + 1} examples in split {split_name}...")
                        translated_splits[split_name] = Dataset.from_list(translated_examples)
                    translated_dataset_dict = DatasetDict(translated_splits)
                    translated_dataset_dict.save_to_disk(translated_output_dir)
                    logging.info(f"Saved translated DatasetDict to {translated_output_dir}")

                elif isinstance(actual_dataset_to_process, Dataset):
                    logging.info(f"Translating single split dataset: {dataset_id}")
                    translated_examples = []
                    text_column = translation_config['text_column']
                    target_lang = translation_config['target_lang']
                    
                    for i, example in enumerate(actual_dataset_to_process):
                        # Convert example to dictionary if it's not already
                        example_dict = dict(example)
                        new_example = dict(example_dict)
                        
                        # Get text using dictionary access
                        original_text = example_dict.get(text_column)
                        
                        if original_text and isinstance(original_text, str):
                            translated_text = api_deepseek_medical_translation(
                                text_to_translate=original_text,
                                source_lang=source_lang,
                                target_lang=target_lang
                            )
                            translation_key = f"translated_{text_column}_{target_lang}"
                            # Use dictionary update instead of direct assignment
                            dict_update(new_example, {
                                translation_key: translated_text if translated_text else "TRANSLATION_FAILED"
                            })
                        elif not original_text:
                            logging.debug(f"Row {i+1}: Column '{text_column}' not found or empty.")
                        elif not isinstance(original_text, str):
                            logging.warning(f"Row {i+1}: Column '{text_column}' is not a string (type: {type(original_text)}). Skipping translation.")
                        translated_examples.append(new_example)
                        if (i + 1) % 100 == 0:
                            logging.info(f"Translated {i + 1} examples in single dataset...")
                    
                    if translated_examples:
                        translated_dataset = Dataset.from_list(translated_examples)
                        translated_dataset.save_to_disk(translated_output_dir) # Saves to a directory, not a single file directly by default
                        logging.info(f"Saved translated single Dataset to {translated_output_dir}")
                else:
                    logging.warning(f"Dataset '{dataset_id}' is not a Dataset or DatasetDict. Skipping translation.")
            # --- End of Added Translation Logic ---

        except Exception as e:
            logging.error(f"Failed to download or save dataset '{dataset_id}': {e}")
            # Clean up potentially incomplete directory
            if os.path.exists(target_dir):
                 try:
                     # Be cautious with recursive deletion, maybe just log the error
                     logging.warning(f"An error occurred. The directory {target_dir} might contain incomplete data.")
                     # import shutil
                     # shutil.rmtree(target_dir)
                     # logging.info(f"Removed potentially incomplete directory: {target_dir}")
                 except Exception as cleanup_e:
                     logging.error(f"Error during cleanup of {target_dir}: {cleanup_e}")
        logging.info(f"--- Finished processing dataset: {dataset_id} ---")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # Handle token saving
    if args.token and args.save_token:
        HfFolder.save_token(args.token)
        logging.info("Hugging Face token saved.")
        # Use the provided token for this run even if saving
        token_to_use = args.token
    elif args.token:
        # Use the provided token without saving it globally
        token_to_use = args.token
    else:
        # Rely on cached token or public access
        token_to_use = None

    download_and_save_datasets(
        dataset_ids=args.datasets,
        base_download_dir=args.download_dir,
        token=token_to_use,
        subset=args.subset
    )
    logging.info("Dataset download process finished.")
