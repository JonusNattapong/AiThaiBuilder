import os
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import logging
import torch
# .env file loading (if needed)
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Typhoon 7B Instruct model
# Determine the base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Define the target directory to save the model
SAVE_DIRECTORY = os.path.join(BASE_PATH, 'model/work', MODEL_NAME.split('/')[-1])

# --- Main Download Logic ---
def download_model():
    """Downloads and saves the specified Hugging Face model and tokenizer."""
    logging.info(f"Attempting to download model: {MODEL_NAME}")
    logging.info(f"Target save directory: {SAVE_DIRECTORY}")

    # Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIRECTORY):
        try:
            os.makedirs(SAVE_DIRECTORY)
            logging.info(f"Created directory: {SAVE_DIRECTORY}")
        except OSError as e:
            logging.error(f"Error creating directory {SAVE_DIRECTORY}: {e}")
            return

    try:
        # Download and load tokenizer with specific configurations
        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        # Save tokenizer
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        logging.info(f"Tokenizer saved successfully to {SAVE_DIRECTORY}")

        # Download and load model with device mapping
        logging.info("Downloading model... (This may take a while)")
        
        # Check available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,  # Always use low memory mode for large models
            torch_dtype=torch.float16,  # Use float16 for efficiency
            offload_folder="offload"  # Enable disk offloading if needed
        )
        
        # Save model
        model.save_pretrained(
            SAVE_DIRECTORY,
            safe_serialization=True
        )
        logging.info(f"Model saved successfully to {SAVE_DIRECTORY}")
        logging.info("Model download and saving complete.")

    except Exception as e:
        logging.error(f"An error occurred during download or saving: {e}")
        logging.error("Please check the model name and your internet connection.")

if __name__ == "__main__":
    download_model()
