import os
import sys
import json
import argparse
import random
from tqdm import tqdm
import re
import csv
import uuid

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

def generate_par_prompt(problem: str):
    """
    Generates prompts for Program-Aided Reasoning (PAR).
    """
    system_prompt = (
        "You are an AI assistant that solves problems by writing and (conceptually) executing Python code. "
        "Explain your reasoning, show the Python code you would use, and then state the result of the code execution and the final answer."
    )
    user_prompt = (
        f"Problem: {problem}\n\n"
        "Please provide a solution that may involve Python code.\n"
        "Format your response as:\n"
        "Reasoning: <Your thought process on how to solve it, including why code might be useful>\n"
        "Python Code:\n```python\n# Your Python code here\n```\n"
        "Execution Result: <The result you would get from running the code>\n"
        "Final Answer: <Your final answer based on the reasoning and code execution>"
    )
    return system_prompt, user_prompt

def process_par_response(problem: str, llm_response: str):
    """
    Processes the LLM's response for PAR data.
    """
    try:
        reasoning_marker = "Reasoning:"
        code_marker_start = "Python Code:\n```python"
        code_marker_end = "```"
        result_marker = "Execution Result:"
        answer_marker = "Final Answer:"

        reasoning = ""
        python_code = ""
        execution_result = ""
        final_answer = ""

        # Extract Reasoning
        re_start = llm_response.find(reasoning_marker)
        code_s_start = llm_response.find(code_marker_start)
        if re_start != -1 and code_s_start != -1:
            reasoning = llm_response[re_start + len(reasoning_marker):code_s_start].strip()
        elif re_start != -1: # if code block is missing but reasoning exists
             reasoning = llm_response[re_start + len(reasoning_marker):].strip().split("\nPython Code:")[0]


        # Extract Python Code
        if code_s_start != -1:
            code_e_start = llm_response.find(code_marker_end, code_s_start + len(code_marker_start))
            if code_e_start != -1:
                python_code = llm_response[code_s_start + len(code_marker_start):code_e_start].strip()
        
        # Extract Execution Result
        res_start = llm_response.find(result_marker)
        ans_start = llm_response.find(answer_marker)
        if res_start != -1 and ans_start != -1:
            execution_result = llm_response[res_start + len(result_marker):ans_start].strip()
        elif res_start != -1: # if answer is missing
            execution_result = llm_response[res_start + len(result_marker):].strip().split("\nFinal Answer:")[0]


        # Extract Final Answer
        if ans_start != -1:
            final_answer = llm_response[ans_start + len(answer_marker):].strip()
        
        # Fallback if parsing is difficult
        if not reasoning and not python_code and not final_answer:
            reasoning = "Parsing Error: Could not extract structured components."

        return {
            "problem": problem,
            "reasoning": reasoning or "N/A",
            "python_code": python_code or "N/A",
            "execution_result": execution_result or "N/A", # This would ideally be from actual execution
            "final_answer": final_answer or "N/A",
            "raw_response": llm_response
        }
    except Exception as e:
        print(f"Error processing PAR response: {e}")
        return {
            "problem": problem,
            "reasoning": "Processing Error",
            "python_code": "Processing Error",
            "execution_result": "Processing Error",
            "final_answer": "Processing Error",
            "raw_response": llm_response
        }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a dataset for Program-Aided Reasoning.
    """
    # TODO: Define problems that benefit from code execution (math, data manipulation, etc.)
    example_problems = [
        "What is the sum of the first 100 positive integers?",
        "Find all prime numbers less than 50.",
        "If a list of numbers is [1, 5, 2, 8, 3], what is the median?",
        "Calculate the factorial of 7."
    ]
    if not example_problems:
        print("Error: No example problems for PAR.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating PAR Samples"):
        problem = random.choice(example_problems)
        
        system_prompt, user_prompt = generate_par_prompt(problem)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_response = generate_with_deepseek(messages, api_key, max_tokens=1000)

        if llm_response and not llm_response.startswith("Error:"):
            processed_sample = process_par_response(problem, llm_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate valid PAR response for problem: {problem}. Error: {llm_response}")
            generated_samples.append({
                "problem": problem,
                "reasoning": "GENERATION_ERROR",
                "python_code": "GENERATION_ERROR",
                "execution_result": "GENERATION_ERROR",
                "final_answer": "GENERATION_ERROR",
                "raw_response": llm_response or "No response"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} PAR samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Program-Aided Reasoning (PAR) dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/par_dataset.jsonl", help="Path to save the dataset.")
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

# สร้างข้อความสำหรับหมวดหมู่ program_aided_reasoning
categories = {
    "program_aided_reasoning": [
        "คำนวณหาพื้นที่ของสามเหลี่ยมที่มีฐานยาว 15.5 เซนติเมตร และสูง 8.2 เซนติเมตร โดยใช้โปรแกรมช่วยคำนวณ",
        "ถ้าอัตราแลกเปลี่ยนปัจจุบันคือ 1 ดอลลาร์สหรัฐ เท่ากับ 36.75 บาท เงิน 550 ดอลลาร์สหรัฐ จะแลกเป็นเงินไทยได้กี่บาท (ใช้โปรแกรมช่วย)",
        "สร้างฟังก์ชันไพทอนเพื่อหาค่าเฉลี่ยของตัวเลขในลิสต์ [10, 25, 30, 45, 60]",
        "แปลงอุณหภูมิจาก 77 องศาฟาเรนไฮต์ เป็นองศาเซลเซียส โดยใช้สูตร C = (F - 32) * 5/9 และให้โปรแกรมช่วยคำนวณ",
        "หากเดินทางด้วยความเร็วเฉลี่ย 80 กิโลเมตรต่อชั่วโมง เป็นระยะทาง 300 กิโลเมตร จะใช้เวลาเดินทางกี่ชั่วโมง (ใช้โปรแกรมช่วยคำนวณ)",
        "เขียนโปรแกรมเพื่อตรวจสอบว่าปี 2024 เป็นปีอธิกสุรทินหรือไม่",
        "คำนวณดอกเบี้ยทบต้นของเงินฝาก 100,000 บาท อัตราดอกเบี้ย 3% ต่อปี เป็นเวลา 5 ปี โดยคิดดอกเบี้ยทบต้นปีละครั้ง (ใช้โปรแกรมช่วย)",
        "สร้างโปรแกรมเพื่อสุ่มตัวเลขระหว่าง 1 ถึง 100 จำนวน 5 ตัวเลข",
        "เขียนโค้ดเพื่อเรียงลำดับรายชื่อต่อไปนี้ตามตัวอักษร: ['สมชาย', 'วิไล', 'อำนาจ', 'พรทิพย์']",
        "คำนวณหาค่าดัชนีมวลกาย (BMI) ของคนที่มีน้ำหนัก 68 กิโลกรัม และส่วนสูง 1.72 เมตร (ใช้โปรแกรมช่วย)"
    ]
}

if __name__ == "__main__":
    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_program_aided_reasoning.csv")

    rows = []
    for label, texts in categories.items():
        for text in texts:
            rows.append([str(uuid.uuid4()), text, label])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")

