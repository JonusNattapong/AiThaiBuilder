import os
from huggingface_hub import HfApi, upload_folder, HfFolder, Repository # Added Repository
import logging
import argparse
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine the base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Command Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Upload a local Hugging Face model to the Hub.")
    parser.add_argument(
        "--local_model_name",
        type=str,
        default="wangchanberta-base-att-spm-uncased", # Default to the previously downloaded model's folder name
        help="The name of the folder inside 'Model' containing the model files."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repository ID on Hugging Face Hub (e.g., 'YourUsername/your-model-name'). Replace 'YourUsername' with your actual HF username."
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model files",
        help="The commit message for the upload."
    )
    parser.add_argument(
        "--create_repo",
        action="store_true",
        help="Create the repository on the Hub if it doesn't exist. Requires appropriate token permissions."
    )
    return parser.parse_args()

# --- Main Upload Logic ---
def upload_model_to_hub(local_model_dir, repo_id, commit_message, create_repo):
    """Uploads the contents of a local directory to a Hugging Face Hub repository."""

    logging.info(f"Attempting to upload model from: {local_model_dir}")
    logging.info(f"Target repository ID on Hub: {repo_id}")

    # Check if the local directory exists
    if not os.path.isdir(local_model_dir):
        logging.error(f"Local model directory not found: {local_model_dir}")
        logging.error("Please ensure the directory exists and contains the model files (config.json, pytorch_model.bin, etc.).")
        return

    try:
        logging.info("Starting upload...")
        # Use upload_folder for simplicity
        upload_folder(
            folder_path=local_model_dir,
            repo_id=repo_id,
            repo_type="model", # Specify repository type as 'model'
            commit_message=commit_message,
            create_pr=False, # Directly commit to main branch
            # The 'create_repo' argument in upload_folder is deprecated,
            # use HfApi().create_repo instead if needed explicitly before upload.
        )

        # Ensure token is loaded (it should be if .env is set and load_dotenv() called)
        token = HfFolder.get_token()
        if not token:
            # Fallback to environment variable if not logged in via CLI
            token = os.environ.get("HF_TOKEN")
            if not token:
                print("Error: Hugging Face token not found.")
                print("Please login using `huggingface-cli login` or set HF_TOKEN in your .env file.")
                return

        print(f"Using Hugging Face token for authentication.")

        logging.info(f"Successfully uploaded contents of {local_model_dir} to {repo_id}")

    except Exception as e:
        logging.error(f"An error occurred during upload: {e}")
        logging.error("Please check your Hugging Face login status (run 'huggingface-cli login'),")
        logging.error("repository ID, network connection, and token permissions.")

if __name__ == "__main__":
    args = parse_args()

    # Construct the full path to the local model directory
    full_local_model_path = os.path.join(BASE_PATH, 'Model', args.local_model_name)

    # --- !!! IMPORTANT !!! ---
    # Ensure the user has replaced the placeholder in repo_id
    if "YourUsername" in args.repo_id:
        logging.error("Please replace 'YourUsername' in the --repo_id argument with your actual Hugging Face username.")
    else:
        upload_model_to_hub(
            local_model_dir=full_local_model_path,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            create_repo=args.create_repo # Note: create_repo flag is mainly informational here
        )
