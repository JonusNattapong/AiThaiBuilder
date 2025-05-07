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
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

def generate_reasoning_prompt(problem_type: str, problem_description: str):
    """
    Generates prompts for general reasoning tasks.
    """
    system_prompt = "You are an AI assistant skilled in various types of reasoning. Solve the given problem and explain your reasoning steps."
    user_prompt = (
        f"Problem Type: {problem_type}\n"
        f"Problem: {problem_description}\n\n"
        "Please provide your reasoning process and the final answer.\n"
        "Format:\n"
        "Reasoning: <your detailed reasoning>\n"
        "Answer: <your final answer>"
    )
    return system_prompt, user_prompt

def process_reasoning_response(problem_type: str, problem_description: str, llm_response: str):
    """
    Processes the LLM's response for general reasoning data.
    """
    try:
        reasoning_marker = "Reasoning:"
        answer_marker = "Answer:"
        
        reasoning_start = llm_response.find(reasoning_marker)
        answer_start = llm_response.find(answer_marker)

        reasoning = ""
        answer = ""

        if reasoning_start != -1 and answer_start != -1:
            reasoning = llm_response[reasoning_start + len(reasoning_marker):answer_start].strip()
            answer = llm_response[answer_start + len(answer_marker):].strip()
        elif reasoning_start != -1: # Only reasoning found
            reasoning = llm_response[reasoning_start + len(reasoning_marker):].strip()
            answer = "Parsing Error: Answer not found"
        else: # Fallback
            reasoning = "Parsing Error: Reasoning not found"
            answer = "Parsing Error: Answer not found"
            if answer_marker in llm_response: # Check if only answer is present
                 answer = llm_response[llm_response.find(answer_marker) + len(answer_marker):].strip()


        return {
            "problem_type": problem_type,
            "problem_description": problem_description,
            "reasoning": reasoning,
            "answer": answer,
            "raw_response": llm_response
        }
    except Exception as e:
        print(f"Error processing reasoning response: {e}")
        return {
            "problem_type": problem_type,
            "problem_description": problem_description,
            "reasoning": "Processing Error",
            "answer": "Processing Error",
            "raw_response": llm_response
        }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a general-purpose reasoning dataset.
    """
    # TODO: Define a diverse set of reasoning problems.
    example_problems = [
        {"type": "Logical Deduction", "description": "All birds can fly. A penguin is a bird. Can a penguin fly? Explain why or why not based *only* on the premises."},
        {"type": "Math Word Problem", "description": "John has 5 apples. He gives 2 to Mary. How many apples does John have left?"},
        {"type": "Causal Inference", "description": "Every time it rains, the grass gets wet. Today, the grass is wet. Did it rain today? Explain your certainty."},
        {"type": "Spatial Reasoning", "description": "A is to the left of B. C is to the right of B. What is the order of A, B, and C from left to right?"}
    ]
    if not example_problems:
        print("Error: No example problems for general reasoning.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating Reasoning Samples"):
        problem_data = random.choice(example_problems)
        problem_type = problem_data["type"]
        problem_description = problem_data["description"]
        
        system_prompt, user_prompt = generate_reasoning_prompt(problem_type, problem_description)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_response = generate_with_deepseek(messages, api_key)

        if llm_response and not llm_response.startswith("Error:"):
            processed_sample = process_reasoning_response(problem_type, problem_description, llm_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate valid reasoning response. Error: {llm_response}")
            generated_samples.append({
                "problem_type": problem_type,
                "problem_description": problem_description,
                "reasoning": "GENERATION_ERROR",
                "answer": "GENERATION_ERROR",
                "raw_response": llm_response or "No response"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} reasoning samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate General Reasoning dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/reasoning_dataset.jsonl", help="Path to save the dataset.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key.")
    
    args = parser.parse_args()
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: DeepSeek API Key not found.")
        return
    generate_dataset(args.output_file, args.num_samples, api_key)

if __name__ == "__main__":
    main()

# สร้างข้อความสำหรับหมวดหมู่ general_reasoning
categories = {
    "general_reasoning": [
        "ถ้าฝนตกหนัก การจราจรมีแนวโน้มจะเป็นอย่างไร เพราะเหตุใด",
        "แมวทุกตัวเป็นสัตว์เลี้ยงลูกด้วยนม สัตว์เลี้ยงลูกด้วยนมทุกตัวหายใจด้วยปอด ดังนั้นแมวหายใจด้วยปอดหรือไม่ เพราะเหตุใด",
        "หากราคาของสินค้า A เพิ่มขึ้น แต่ความต้องการสินค้า A ไม่เปลี่ยนแปลง ยอดขายของสินค้า A จะเป็นอย่างไร",
        "นาย ก วิ่งเร็วกว่านาย ข แต่นาย ข วิ่งเร็วกว่านาย ค ใครวิ่งช้าที่สุด",
        "ถ้าวันนี้เป็นวันพุธ อีก 3 วันข้างหน้าจะเป็นวันอะไร",
        "ในตะกร้ามีผลไม้ 3 ชนิด คือ ส้ม แอปเปิ้ล และกล้วย หากหยิบผลไม้มา 1 ชิ้นโดยไม่มอง โอกาสที่จะหยิบได้ส้มเป็นเท่าใด ถ้ามีส้ม 5 ลูก แอปเปิ้ล 3 ลูก และกล้วย 2 ลูก",
        "ถ้า 'นก' สัมพันธ์กับ 'บิน' แล้ว 'ปลา' สัมพันธ์กับอะไร",
        "หากต้องการเดินทางจากกรุงเทพไปเชียงใหม่ วิธีใดใช้เวลาน้อยที่สุดระหว่างเครื่องบิน รถไฟ และรถยนต์ส่วนตัว เพราะเหตุใด",
        "ถ้า A=1, B=2, C=3 แล้วคำว่า 'CAB' จะมีค่าเท่ากับเท่าไร",
        "ทำไมน้ำแข็งจึงลอยน้ำได้"
    ]
}

if __name__ == "__main__":
    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_general_reasoning.csv")

    rows = []
    for label, texts in categories.items():
        for text in texts:
            rows.append([str(uuid.uuid4()), text, label])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")

