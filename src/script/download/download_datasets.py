# -*- coding: utf-8 -*-
"""
Script to download datasets from Hugging Face Hub

Example Usage:
python download_hf_datasets.py --datasets USERNAME/DATASET1 USERNAME/DATASET2
python download_hf_datasets.py -d USERNAME/DATASET --subset configuration_name
"""

import os
import logging
import argparse
from datasets import load_dataset, DownloadConfig, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from huggingface_hub import HfFolder

# --- Configuration ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'DatasetDownload')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub.")
    parser.add_argument(
        "-d", "--datasets",
        nargs='+',
        required=True,
        help="List of dataset IDs from Hugging Face Hub (e.g., 'username/dataset_name')."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Base directory to save datasets. Default: {DEFAULT_DOWNLOAD_DIR}"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Hub token for private datasets."
    )
    parser.add_argument(
        "--save_token",
        action="store_true",
        help="Save the provided token for future use."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Specific subset/configuration of the dataset to download."
    )
    return parser.parse_args()

# --- Download Logic ---
def download_datasets(dataset_ids, base_download_dir, token=None, subset=None):
    """Download datasets from Hugging Face Hub."""
    logging.info(f"Base download directory: {base_download_dir}")
    os.makedirs(base_download_dir, exist_ok=True)

    download_config = DownloadConfig(token=token)

    for dataset_id in dataset_ids:
        logging.info(f"Processing dataset: {dataset_id}")
        safe_dir_name = dataset_id.replace('/', '_')
        target_dir = os.path.join(base_download_dir, safe_dir_name)

        if os.path.exists(target_dir):
            logging.warning(f"Directory exists: {target_dir}. Skipping.")
            continue

        try:
            # Load dataset in streaming mode first to check splits
            streaming_dataset = load_dataset(
                dataset_id,
                name=subset,
                download_config=download_config,
                streaming=True
            )
            
            # Load regular dataset for processing
            dataset = load_dataset(
                dataset_id,
                name=subset,
                download_config=download_config
            )

            # Save dataset
            if isinstance(dataset, (Dataset, DatasetDict)):
                dataset.save_to_disk(target_dir)
                logging.info(f"Saved dataset to {target_dir}")
            elif isinstance(dataset, (IterableDataset, IterableDatasetDict)):
                converted = DatasetDict()
                splits_to_process = ['train'] if isinstance(dataset, IterableDataset) else list(streaming_dataset.keys())
                
                for split_name in splits_to_process:
                    examples = []
                    split_data = streaming_dataset[split_name] if isinstance(streaming_dataset, IterableDatasetDict) else streaming_dataset
                    for example in split_data:
                        examples.append(example)
                    converted[split_name] = Dataset.from_list(examples)
                
                converted.save_to_disk(target_dir)
                logging.info(f"Converted and saved iterable dataset to {target_dir}")

        except Exception as e:
            logging.error(f"Error processing {dataset_id}: {e}")
            if os.path.exists(target_dir):
                logging.warning(f"Partial data may exist in {target_dir}")

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    if args.token and args.save_token:
        HfFolder.save_token(args.token)
        logging.info("Hugging Face token saved.")
    
    download_datasets(
        dataset_ids=args.datasets,
        base_download_dir=args.download_dir,
        token=args.token,
        subset=args.subset
    )
    
    logging.info("Dataset download completed.")