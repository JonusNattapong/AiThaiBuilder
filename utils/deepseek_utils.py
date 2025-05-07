import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def get_deepseek_api_key(api_key_input=None):
    if api_key_input and api_key_input.strip(): # Check if api_key_input is provided and not empty
        return api_key_input
    env_api_key = os.getenv("DEEPSEEK_API_KEY")
    if env_api_key and env_api_key.strip() and env_api_key != "your_actual_deepseek_api_key_here":
        return env_api_key
    return None # Return None if no valid key is found

def generate_with_deepseek(prompt_messages, api_key_val, model="deepseek-chat", max_tokens=1024, temperature=0.7):
    """
    Generates text using the Deepseek API.

    Args:
        prompt_messages (list): A list of message objects (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
        api_key_val (str): The Deepseek API key.
        model (str): The model to use (e.g., "deepseek-chat").
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        str: The generated text, or an error message string.
    """
    if not api_key_val:
        return "Error: Deepseek API key not provided or invalid. Please check UI input or .env file."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_val}"
    }

    payload = {
        "model": model,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False 
    }

    response = None
    result = None
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120) # Increased timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0 and result["choices"][0].get("message") and result["choices"][0]["message"].get("content"):
            return result["choices"][0]["message"]["content"].strip()
        
        # If we get here, the response doesn't have the expected content
        error_detail = result.get("error", {}).get("message", "Unknown API response structure.")
        return f"Error: No content in API response. Details: {error_detail}. Response: {json.dumps(result)}"
            
    except requests.exceptions.HTTPError as e:
        # Attempt to get more specific error from response body for 4xx/5xx errors
        error_body = e.response.text if e.response else "No response available"
        try:
            error_json = e.response.json() if e.response else {}
            error_message = error_json.get("error", {}).get("message", error_body)
        except (json.JSONDecodeError, AttributeError):
            error_message = error_body
        return f"Error: HTTP {e.response.status_code if e.response else 'Unknown'} calling Deepseek API: {error_message}"
    
    except requests.exceptions.RequestException as e:
        return f"Error: Network or request issue calling Deepseek API: {str(e)}"
    
    except json.JSONDecodeError:
        response_text = response.text if response else "No response available"
        return f"Error: Could not decode JSON response from API. Response text: {response_text}"
    
    except KeyError:
        result_text = json.dumps(result) if result else "No result available"
        response_text = response.text if response else "No response available"
        return f"Error: Unexpected API response format. Result: {result_text}, Response: {response_text}"
    
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


if __name__ == '__main__':
    # Example usage (requires DEEPSEEK_API_KEY in .env or passed directly)
    # Create a .env file with your DEEPSEEK_API_KEY if you want to test this directly
    # DEEPSEEK_API_KEY="your_actual_api_key"
    
    print("Testing Deepseek API utility...")
    test_api_key = get_deepseek_api_key() 
    
    if not test_api_key:
        print("Deepseek API key not found. Please set DEEPSEEK_API_KEY in .env file or provide it for testing.")
    else:
        print(f"Using API Key: {'*' * (len(test_api_key) - 4) + test_api_key[-4:]}")
        
        print("\nTest 1: Simple greeting")
        messages_greeting = [
            {"role": "system", "content": "You are a helpful assistant that speaks Thai."},
            {"role": "user", "content": "สวัสดีตอนเช้า"}
        ]
        generated_text_greeting = generate_with_deepseek(messages_greeting, test_api_key)
        print("Generated Text (Greeting):")
        print(generated_text_greeting)

        print("\nTest 2: Thai text generation about cats")
        messages_cats = [
            {"role": "system", "content": "คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญในการสร้างข้อมูลภาษาไทย"},
            {"role": "user", "content": "สร้างข้อความสั้นๆ เกี่ยวกับแมวไทย"}
        ]
        generated_text_cats = generate_with_deepseek(messages_cats, test_api_key)
        print("Generated Thai Text (Cats):")
        print(generated_text_cats)

        print("\nTest 3: Invalid API Key (simulated by passing a known bad key if you have one, or None)")
        # To truly test this, you might pass an invalid key string.
        # For now, we'll test the None case if no key was found initially.
        if not test_api_key: # If key wasn't found, this will test the error path
             error_text = generate_with_deepseek(messages_cats, None)
             print("Generated Text (Invalid Key - None):")
             print(error_text)
        else: # If a key was found, simulate passing an invalid one
            invalid_key = "sk-invalidkey123"
            print(f"Simulating with an invalid key: {invalid_key[:7]}...")
            error_text_invalid = generate_with_deepseek(messages_cats, invalid_key)
            print("Generated Text (Invalid Key - Simulated):")
            print(error_text_invalid)

        print("\nAll tests complete.")
