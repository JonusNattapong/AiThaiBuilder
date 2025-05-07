import os
import sys
import json
import argparse
import random
from tqdm import tqdm
import csv
import uuid

# Add project root to sys.path to allow importing from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

def generate_cot_prompt(problem: str):
    """
    Generates system and user prompts for Chain of Thought reasoning.
    Customize this function to define your problem types and CoT elicitation.
    """
    system_prompt = "You are an AI assistant that thinks step by step to solve problems. Explain your reasoning clearly."
    user_prompt = (
        f"Problem: {problem}\n\n"
        "Please provide a step-by-step solution (chain of thought) and then the final answer.\n"
        "Format your response as:\n"
        "Chain of Thought: <your detailed step-by-step reasoning>\n"
        "Final Answer: <your final answer>"
    )
    return system_prompt, user_prompt

def process_cot_response(problem: str, llm_response: str):
    """
    Processes the LLM's response to extract the chain of thought and final answer.
    Customize this based on the expected output format from your prompt.
    """
    try:
        # Basic parsing, assuming the LLM follows the format.
        # More robust parsing might be needed.
        thought_marker = "Chain of Thought:"
        answer_marker = "Final Answer:"
        
        thought_start = llm_response.find(thought_marker)
        answer_start = llm_response.find(answer_marker)

        if thought_start != -1 and answer_start != -1:
            chain_of_thought = llm_response[thought_start + len(thought_marker):answer_start].strip()
            final_answer = llm_response[answer_start + len(answer_marker):].strip()
            return {
                "problem": problem,
                "chain_of_thought": chain_of_thought,
                "final_answer": final_answer,
                "raw_response": llm_response
            }
        elif thought_start != -1: # Only thought found
             chain_of_thought = llm_response[thought_start + len(thought_marker):].strip()
             return {
                "problem": problem,
                "chain_of_thought": chain_of_thought,
                "final_answer": "Parsing Error: Answer not found",
                "raw_response": llm_response
            }
        else:
            # Fallback if markers are not found
            return {
                "problem": problem,
                "chain_of_thought": "Parsing Error: CoT not found",
                "final_answer": "Parsing Error: Answer not found",
                "raw_response": llm_response
            }
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return {
            "problem": problem,
            "chain_of_thought": "Processing Error",
            "final_answer": "Processing Error",
            "raw_response": llm_response
        }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a dataset for Chain of Thought reasoning.
    """
    # TODO: Replace with your actual problem generation logic or load problems from a file.
    example_problems = [
        "If a train travels at 60 km/h for 3 hours, how far does it travel?",
        "What is the capital of France?",
        "Explain the process of photosynthesis in simple terms.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    ]

    if not example_problems:
        print("Error: No example problems provided. Please define 'example_problems'.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating CoT Samples"):
        problem = random.choice(example_problems) # Select a random problem
        
        system_prompt, user_prompt = generate_cot_prompt(problem)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_response = generate_with_deepseek(messages, api_key)

        if llm_response and not llm_response.startswith("Error:"):
            processed_sample = process_cot_response(problem, llm_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate valid response for problem: {problem}. Error: {llm_response}")
            generated_samples.append({
                "problem": problem,
                "chain_of_thought": "GENERATION_ERROR",
                "final_answer": "GENERATION_ERROR",
                "raw_response": llm_response or "No response"
            })

    # Save to JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} CoT samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Chain of Thought (CoT) dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/cot_dataset.jsonl", help="Path to save the generated dataset.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    # API key can also be sourced from .env file by get_deepseek_api_key
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key (optional, will use .env if not provided).")
    
    args = parser.parse_args()

    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: DeepSeek API Key not found. Please provide it via --api_key or set DEEPSEEK_API_KEY in your .env file.")
        return

    generate_dataset(args.output_file, args.num_samples, api_key)

    # Generate CSV file for Thai text prompts
    categories = {
        "chain_of_thought": [
            "อธิบายขั้นตอนการวางแผนการเดินทางไปต่างประเทศครั้งแรก",
            "ถ้าต้องการลดน้ำหนัก 5 กิโลกรัมภายใน 1 เดือน ควรทำอย่างไรบ้าง แสดงเหตุผลทีละขั้นตอน",
            "นักเรียนคนหนึ่งต้องการเพิ่มเกรดวิชาคณิตศาสตร์จาก C เป็น A เขาควรวางแผนการเรียนอย่างไรบ้าง",
            "หากต้องการเปิดร้านกาแฟเล็กๆ ต้องเตรียมตัวและดำเนินการอย่างไรบ้าง ให้บอกเป็นลำดับขั้น",
            "อธิบายกระบวนการตัดสินใจเลือกซื้อคอมพิวเตอร์เครื่องใหม่ โดยพิจารณาจากปัจจัยต่างๆ",
            "ถ้าเกิดเหตุการณ์ไฟไหม้ในอาคารสูง ควรปฏิบัติตนอย่างไรเพื่อความปลอดภัย แสดงเป็นขั้นตอน",
            "วางแผนการจัดงานวันเกิดให้เพื่อนสนิท โดยมีงบประมาณจำกัด ต้องทำอะไรบ้าง",
            "อธิบายวิธีแก้ปัญหาเมื่อรถยนต์สตาร์ทไม่ติดเบื้องต้น ควรตรวจสอบอะไรบ้างตามลำดับ",
            "ถ้าต้องการเรียนรู้ทักษะการเขียนโปรแกรม ควรเริ่มต้นอย่างไร และมีขั้นตอนการพัฒนาตนเองอย่างไร",
            "อธิบายกระบวนการยื่นภาษีเงินได้บุคคลธรรมดาผ่านระบบออนไลน์ทีละขั้นตอน"
        ]
    }

    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_chain_of_thought.csv")

    rows = []
    for label, texts in categories.items():
        for text in texts:
            rows.append([str(uuid.uuid4()), text, label])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")

if __name__ == "__main__":
    main()

