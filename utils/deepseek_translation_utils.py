import os
import sys

# Attempt to import from the .deepseek_utils module within the same package
# This is the standard way if 'utils' is treated as a package.
try:
    from .deepseek_utils import generate_with_deepseek, get_deepseek_api_key
except ImportError:
    # Fallback for scenarios where the script might be run directly,
    # or the Python path isn't set up to recognize 'utils' as a package immediately.
    # This adds the project root (parent directory of 'utils') to sys.path
    # to allow finding the 'utils' package for the import.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    # Now try importing again, assuming 'utils' is a package in the project root
    from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key


def get_translation_prompt(source_text, source_lang_code, target_lang_code, additional_instructions=""):
    """
    Generates a standardized prompt for translation tasks.
    """
    source_lang_map = {
        "EN": "English",
        "ZH": "Chinese (Simplified)"
    }
    target_lang_map = {
        "TH": "Thai"
    }

    source_language = source_lang_map.get(source_lang_code.upper(), source_lang_code)
    target_language = target_lang_map.get(target_lang_code.upper(), target_lang_code)

    system_prompt = f"You are an expert linguist and translator specializing in translating text from {source_language} to {target_language}. Translate accurately, naturally, and maintain the original meaning and tone."

    user_prompt = f"Please translate the following {source_language} text to {target_language}:\n\n"
    user_prompt += f"{source_language} Text:\n\"\"\"\n{source_text}\n\"\"\"\n\n"
    user_prompt += f"{target_language} Translation:"
    if additional_instructions:
        user_prompt += f"\n\nAdditional Instructions: {additional_instructions}"

    return system_prompt, user_prompt

def generate_translation_sample(source_text, source_lang_code, target_lang_code, api_key, additional_instructions=""):
    """
    Generates a single translation sample using the Deepseek API.

    Args:
        source_text (str): The text to be translated.
        source_lang_code (str): The source language code (e.g., "EN", "ZH").
        target_lang_code (str): The target language code (e.g., "TH").
        api_key (str): The Deepseek API key.
        additional_instructions (str, optional): Any additional instructions for the translation.

    Returns:
        str: The translated text, or an error message.
    """
    if not source_text:
        return "Error: Source text cannot be empty."
    if not source_lang_code or not target_lang_code:
        return "Error: Source and target language codes must be specified."

    system_prompt, user_prompt = get_translation_prompt(source_text, source_lang_code, target_lang_code, additional_instructions)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    translated_text = generate_with_deepseek(messages, api_key)
    return translated_text

def generate_en_to_th_translation(english_text, api_key, additional_instructions=""):
    """Generates an English to Thai translation."""
    return generate_translation_sample(english_text, "EN", "TH", api_key, additional_instructions)

def generate_zh_to_th_translation(chinese_text, api_key, additional_instructions=""):
    """Generates a Chinese (Simplified) to Thai translation."""
    return generate_translation_sample(chinese_text, "ZH", "TH", api_key, additional_instructions)

if __name__ == '__main__':
    print("Testing Deepseek Translation Utilities...")

    # This section is for standalone testing of this script.
    # It ensures that the .env file in the project root is loaded for API keys.
    current_script_path = os.path.abspath(__file__)
    # utils_dir = os.path.dirname(current_script_path) # This is utils/
    project_root_path = os.path.dirname(os.path.dirname(current_script_path)) # This should be RunThaiGenDataset/

    # Ensure project root is in sys.path for imports like `from dotenv import load_dotenv`
    # and for `get_deepseek_api_key` to potentially find .env if its own `load_dotenv` call needs it.
    original_sys_path = list(sys.path) # Store original sys.path
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)

    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(project_root_path, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            print(f"Loaded .env from: {dotenv_path}")
        else:
            print(f".env file not found at: {dotenv_path}. API key must be available in environment or passed to functions.")
    except ImportError:
        print("python-dotenv library not found. API key must be available in environment or passed to functions.")
    except Exception as e:
        print(f"Error loading .env: {e}")

    # get_deepseek_api_key itself calls load_dotenv(), but the path adjustment above helps ensure it finds the .env file
    # correctly when this script is run directly from the utils directory.
    api_key_to_test = get_deepseek_api_key()

    if not api_key_to_test:
        print("Deepseek API key not found. Please ensure DEEPSEEK_API_KEY is set in your .env file in the project root or as an environment variable.")
    else:
        print(f"Using API Key: {'*' * (len(api_key_to_test) - 4) + api_key_to_test[-4:]}")

        # Test EN to TH
        print("\n--- Testing English to Thai Translation ---")
        en_sample_text = "Hello, how are you today? I hope you are doing well."
        print(f"Original English: {en_sample_text}")
        thai_translation_en = generate_en_to_th_translation(en_sample_text, api_key_to_test, additional_instructions="Use a friendly tone.")
        print(f"Thai Translation: {thai_translation_en}")

        en_sample_text_2 = "The weather is beautiful and the birds are singing."
        print(f"\nOriginal English: {en_sample_text_2}")
        thai_translation_en_2 = generate_en_to_th_translation(en_sample_text_2, api_key_to_test)
        print(f"Thai Translation: {thai_translation_en_2}")

        # Test ZH to TH
        print("\n--- Testing Chinese (Simplified) to Thai Translation ---")
        zh_sample_text = "你好，你今天过得怎么样？我希望你一切都好。" # Hello, how are you today? I hope you are doing well.
        print(f"Original Chinese: {zh_sample_text}")
        thai_translation_zh = generate_zh_to_th_translation(zh_sample_text, api_key_to_test, additional_instructions="ใช้คำสุภาพ") # Use polite words
        print(f"Thai Translation: {thai_translation_zh}")

        zh_sample_text_2 = "今天天气很好，鸟儿在唱歌。" # The weather is beautiful today and the birds are singing.
        print(f"\nOriginal Chinese: {zh_sample_text_2}")
        thai_translation_zh_2 = generate_zh_to_th_translation(zh_sample_text_2, api_key_to_test)
        print(f"Thai Translation: {thai_translation_zh_2}")

        print("\nTranslation utility tests complete.")

    # Restore original sys.path if it was modified
    sys.path = original_sys_path
