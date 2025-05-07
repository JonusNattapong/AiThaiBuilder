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

def generate_initial_solution_prompt(problem: str):
    system_prompt = "You are an AI assistant. Solve the following problem."
    user_prompt = f"Problem: {problem}\n\nProvide your solution:"
    return system_prompt, user_prompt

def generate_reflection_prompt(problem: str, initial_solution: str):
    """
    Generates prompts for reflection on an initial solution.
    """
    system_prompt = (
        "You are an AI assistant that reviews and improves solutions. "
        "Critically evaluate the provided solution and then offer a revised, improved solution."
    )
    user_prompt = (
        f"Problem: {problem}\n\n"
        f"Initial Solution Provided:\n\"\"\"\n{initial_solution}\n\"\"\"\n\n"
        "Please perform the following steps:\n"
        "1. Critique the initial solution: Identify any errors, areas for improvement, or alternative approaches.\n"
        "2. Revised Solution: Provide an improved and more accurate solution based on your critique.\n\n"
        "Format your response as:\n"
        "Critique: <your critique of the initial solution>\n"
        "Revised Solution: <your improved solution>"
    )
    return system_prompt, user_prompt

def process_reflection_response(problem: str, initial_solution: str, llm_reflection_response: str):
    """
    Processes the LLM's reflection response.
    """
    try:
        critique_marker = "Critique:"
        revised_solution_marker = "Revised Solution:"
        
        critique_start = llm_reflection_response.find(critique_marker)
        revised_start = llm_reflection_response.find(revised_solution_marker)

        critique = ""
        revised_solution = ""

        if critique_start != -1 and revised_start != -1:
            critique = llm_reflection_response[critique_start + len(critique_marker):revised_start].strip()
            revised_solution = llm_reflection_response[revised_start + len(revised_solution_marker):].strip()
        elif critique_start != -1: # Only critique found
            critique = llm_reflection_response[critique_start + len(critique_marker):].strip()
            revised_solution = "Parsing Error: Revised solution not found"
        else: # Fallback
            critique = "Parsing Error: Critique not found"
            revised_solution = "Parsing Error: Revised solution not found"
            if revised_solution_marker in llm_reflection_response:
                revised_solution = llm_reflection_response[llm_reflection_response.find(revised_solution_marker) + len(revised_solution_marker):].strip()


        return {
            "problem": problem,
            "initial_solution": initial_solution,
            "critique": critique,
            "revised_solution": revised_solution,
            "raw_reflection_response": llm_reflection_response
        }
    except Exception as e:
        print(f"Error processing reflection response: {e}")
        return {
            "problem": problem,
            "initial_solution": initial_solution,
            "critique": "Processing Error",
            "revised_solution": "Processing Error",
            "raw_reflection_response": llm_reflection_response
        }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a dataset for reflection.
    Self-consistency (generating multiple diverse paths and voting) is more complex
    and not fully implemented here. This focuses on the reflection step.
    """
    # TODO: Define problems where reflection can lead to improvement.
    example_problems = [
        "Write a short story about a time-traveling cat. The story should have a surprising twist.",
        "Explain the concept of black holes to a 10-year-old.",
        "Outline a plan to reduce plastic waste in your local community."
    ]
    if not example_problems:
        print("Error: No example problems for reflection.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating Reflection Samples"):
        problem = random.choice(example_problems)
        
        # Step 1: Generate initial solution
        s_prompt_initial, u_prompt_initial = generate_initial_solution_prompt(problem)
        messages_initial = [{"role": "system", "content": s_prompt_initial}, {"role": "user", "content": u_prompt_initial}]
        initial_solution = generate_with_deepseek(messages_initial, api_key)

        if not initial_solution or initial_solution.startswith("Error:"):
            print(f"Warning: Failed to generate initial solution for: {problem}. Error: {initial_solution}")
            generated_samples.append({
                "problem": problem, "initial_solution": "GENERATION_ERROR", 
                "critique": "N/A", "revised_solution": "N/A",
                "raw_reflection_response": "N/A due to initial solution error"
            })
            continue
            
        # Step 2: Generate reflection and revised solution
        s_prompt_reflect, u_prompt_reflect = generate_reflection_prompt(problem, initial_solution)
        messages_reflect = [{"role": "system", "content": s_prompt_reflect}, {"role": "user", "content": u_prompt_reflect}]
        reflection_response = generate_with_deepseek(messages_reflect, api_key, max_tokens=1200)

        if reflection_response and not reflection_response.startswith("Error:"):
            processed_sample = process_reflection_response(problem, initial_solution, reflection_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate reflection for: {problem}. Error: {reflection_response}")
            generated_samples.append({
                "problem": problem,
                "initial_solution": initial_solution,
                "critique": "GENERATION_ERROR",
                "revised_solution": "GENERATION_ERROR",
                "raw_reflection_response": reflection_response or "No response"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} reflection samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Reflection dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/reflection_dataset.jsonl", help="Path to save the dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key.")
    
    args = parser.parse_args()
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: DeepSeek API Key not found.")

# สร้างข้อความสำหรับหมวดหมู่ reflection_self_consistency
categories = {
    "reflection_self_consistency": [
        "คำถาม: เมืองหลวงของออสเตรเลียคืออะไร? ตอบเบื้องต้น: ซิดนีย์ --- ทบทวนคำตอบ: ซิดนีย์เป็นเมืองใหญ่ แต่เมืองหลวงจริงๆ คืออะไร ลองตรวจสอบอีกครั้ง",
        "แก้โจทย์คณิตศาสตร์: (15 + 5) * 2 - 10 = ? ลองคิดหลายๆ วิธีเพื่อให้ได้คำตอบที่ถูกต้องและตรวจสอบความสอดคล้องกัน",
        "เขียนสรุปบทความเกี่ยวกับภาวะโลกร้อนความยาว 1 ย่อหน้า จากนั้นลองอ่านทบทวนและปรับปรุงให้กระชับและครอบคลุมยิ่งขึ้น",
        "วางแผนการเดินทาง 3 วัน 2 คืนที่ภูเก็ตด้วยงบ 5,000 บาท ลองร่างแผนคร่าวๆ แล้วพิจารณาความเป็นไปได้และปรับแก้",
        "ให้เหตุผลว่าทำไมการออกกำลังกายจึงสำคัญต่อสุขภาพ (ลองให้เหตุผลหลายๆ ข้อ แล้วเลือกข้อที่ดีที่สุด)",
        "แปลประโยคนี้เป็นภาษาอังกฤษ: 'แมวกินปลา' ลองแปลหลายๆ แบบแล้วเลือกแบบที่เป็นธรรมชาติที่สุด",
        "ปัญหา: ลูกค้าไม่พอใจกับสินค้าที่ซื้อไป ควรตอบอีเมลลูกค้าอย่างไร? ลองร่างคำตอบหลายๆ แบบแล้วเลือกแบบที่เหมาะสมที่สุด",
        "วิเคราะห์ข้อดีข้อเสียของการทำงานจากที่บ้าน (Work From Home) ลองลิสต์ออกมาหลายๆ ด้านแล้วสรุป",
        "ถ้ามีคนถามว่า 'อะไรคือสิ่งที่สำคัญที่สุดในชีวิต?' คุณจะตอบว่าอย่างไร ลองคิดคำตอบหลายๆ แง่มุม",
        "เขียนเรื่องสั้นในหัวข้อ 'การผจญภัยในป่าใหญ่' จากนั้นลองอ่านและแก้ไขสำนวนภาษาให้สละสลวยขึ้น"
    ]
}

if __name__ == "__main__":
    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_reflection_self_consistency.csv")

    rows = []
    for label, texts in categories.items():
        for text in texts:
            rows.append([str(uuid.uuid4()), text, label])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")