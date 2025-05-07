import os
import sys
import json
import argparse
import random
from tqdm import tqdm
import csv
import uuid

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if (project_root_dir not in sys.path):
    sys.path.insert(0, project_root_dir)

from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

# Define a list of available tools for the LLM to "know" about
# For actual Toolformer, these tools would have real implementations.
AVAILABLE_TOOLS_DESCRIPTIONS = [
    "Calculator[expression] - Solves mathematical expressions. E.g., Calculator[2*5+3].",
    "Search[query] - Searches for information on the web. E.g., Search[current weather in Paris].",
    "Calendar[date_query] - Checks calendar events. E.g., Calendar[today's appointments]."
]

def generate_toolformer_prompt(text_scenario: str):
    """
    Generates prompts for Toolformer-style data.
    The LLM is asked to rewrite text to include API calls where appropriate.
    """
    tools_list_str = "\n".join([f"- {tool}" for tool in AVAILABLE_TOOLS_DESCRIPTIONS])
    system_prompt = (
        "You are an AI assistant that augments text by inserting API calls to external tools where they would be beneficial. "
        "The available tools are:\n"
        f"{tools_list_str}\n"
        "When you identify a place in the text where a tool could provide useful information, "
        "insert the API call in the format ToolName[input]. Then, provide a hypothetical API_RESPONSE(...) and continue the text incorporating that response."
    )
    user_prompt = (
        f"Consider the following text scenario:\n\"\"\"\n{text_scenario}\n\"\"\"\n\n"
        "Please rewrite this text, inserting API calls (e.g., Search[query], Calculator[expression]) "
        "where they would be useful to obtain information or perform a calculation. "
        "After an API_CALL, include a hypothetical API_RESPONSE(...) and then continue the text naturally, "
        "incorporating the information from the API_RESPONSE.\n\n"
        "Example:\n"
        "Original: I need to know the capital of Italy and then add 5 to 10.\n"
        "Rewritten: I need to know the capital of Italy. API_CALL(Search[capital of Italy]) API_RESPONSE(Rome) So, the capital is Rome. "
        "Then I need to add 5 to 10. API_CALL(Calculator[5+10]) API_RESPONSE(15) The sum is 15.\n\n"
        "Your rewritten text:"
    )
    return system_prompt, user_prompt

def process_toolformer_response(original_scenario: str, llm_response: str):
    """
    Processes the LLM's response for Toolformer-style data.
    """
    # The llm_response itself is the text with API calls.
    return {
        "original_scenario": original_scenario,
        "text_with_api_calls": llm_response, # This is the target output
        "raw_response": llm_response
    }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a dataset for Toolformer-style learning.
    """
    # TODO: Define text scenarios where tool use would be natural.
    example_scenarios = [
        "I wonder what the weather is like in Tokyo today. After that, I need to figure out how much a 15% tip on a $50 bill would be.",
        "To plan my trip, first I need to find out how many kilometers are in 10 miles. Then, I need to check if there are any public holidays in France next week.",
        "What is the population of Canada? And what is 2 to the power of 10?"
    ]
    if not example_scenarios:
        print("Error: No example scenarios for Toolformer.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating Toolformer Samples"):
        scenario = random.choice(example_scenarios)
        
        system_prompt, user_prompt = generate_toolformer_prompt(scenario)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_response = generate_with_deepseek(messages, api_key, max_tokens=1000)

        if llm_response and not llm_response.startswith("Error:"):
            processed_sample = process_toolformer_response(scenario, llm_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate valid Toolformer response for scenario: {scenario}. Error: {llm_response}")
            generated_samples.append({
                "original_scenario": scenario,
                "text_with_api_calls": "GENERATION_ERROR",
                "raw_response": llm_response or "No response"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} Toolformer samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Toolformer-style dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/toolformer_dataset.jsonl", help="Path to save the dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key.")
    
    args = parser.parse_args()
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: DeepSeek API Key not found.")
        return
    generate_dataset(args.output_file, args.num_samples, api_key)

if __name__ == "__main__":
    main()

# สร้างข้อความสำหรับหมวดหมู่ toolformer
categories = {
    "toolformer": [
        "คำนวณ 5 ยกกำลัง 3 ได้ผลลัพธ์เท่าไร", # Calculator
        "พยากรณ์อากาศสำหรับเมืองเชียงใหม่ในวันพรุ่งนี้เป็นอย่างไร", # Weather API
        "ค้นหาบทความเกี่ยวกับประโยชน์ของการทำสมาธิ", # Search Engine
        "แปลคำว่า 'ขอบคุณ' เป็นภาษาญี่ปุ่น", # Translation API
        "วันนี้วันที่เท่าไหร่ เดือนอะไร ปีอะไร", # Calendar API
        "100 ดอลลาร์สหรัฐ เท่ากับกี่บาทไทยในอัตราแลกเปลี่ยนปัจจุบัน", # Currency Converter API
        "ใครเป็นผู้แต่งนวนิยายเรื่อง 'แฮร์รี่ พอตเตอร์'?", # Knowledge Base/Search
        "ระยะทางจากกรุงเทพถึงภูเก็ตประมาณกี่กิโลเมตร", # Maps API
        "เพิ่มรายการ 'ซื้อนม' ในรายการสิ่งที่ต้องทำของฉัน", # To-do List API
        "เล่นเพลง 'Let It Be' ของ The Beatles", # Music Player API
        "สรุปข่าวเด่นประจำวันนี้เกี่ยวกับเทคโนโลยี AI", # News API
        "ค้นหาร้านอาหารไทยใกล้ฉันที่เปิดให้บริการอยู่ตอนนี้" # Places API
    ]
}

if __name__ == "__main__":
    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_toolformer.csv")

    rows = []
    for label, texts in categories.items():
        for text in texts:
            rows.append([str(uuid.uuid4()), text, label])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")

