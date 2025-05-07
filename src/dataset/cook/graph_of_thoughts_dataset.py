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

def generate_got_prompt(problem: str):
    """
    Generates prompts for Graph of Thoughts (GoT) style reasoning.
    This is a simplified version. True GoT involves complex graph operations.
    This prompt aims to elicit a structured exploration of thoughts.
    """
    system_prompt = (
        "You are an AI assistant that explores multiple reasoning paths to solve complex problems. "
        "Structure your thoughts, consider alternatives, and evaluate them."
    )
    user_prompt = (
        f"Problem: {problem}\n\n"
        "Please explore different approaches or thoughts to solve this problem. For each main thought, "
        "consider its implications or next steps. You can structure this as a list of thoughts, "
        "where each thought might have sub-thoughts or evaluations.\n"
        "Example Structure:\n"
        "Thought 1: <Initial idea/approach>\n"
        "  - Evaluation: <Pros/cons of Thought 1>\n"
        "  - Next Step/Sub-Thought: <If Thought 1 is pursued, what's next?>\n"
        "Thought 2: <Alternative idea/approach>\n"
        "  - Evaluation: <Pros/cons of Thought 2>\n"
        "Finally, provide a concluding thought or solution based on your exploration."
    )
    return system_prompt, user_prompt

def process_got_response(problem: str, llm_response: str):
    """
    Processes the LLM's response for GoT-style data.
    """
    # For this simplified version, we'll store the raw exploration.
    # True GoT would require parsing a graph structure.
    return {
        "problem": problem,
        "thought_exploration": llm_response, # The raw structured response from the LLM
        "raw_response": llm_response
    }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a dataset for Graph of Thoughts reasoning.
    """
    # TODO: Define complex problems suitable for GoT-style exploration.
    example_problems = [
        "Plan a 3-day trip to a new city with a budget of $500, considering sightseeing, food, and accommodation.",
        "Outline different strategies for a company to increase its market share in a competitive industry.",
        "What are the potential long-term societal impacts of widespread AI adoption?"
    ]
    if not example_problems:
        print("Error: No example problems provided for GoT.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating GoT Samples"):
        problem = random.choice(example_problems)
        
        system_prompt, user_prompt = generate_got_prompt(problem)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_response = generate_with_deepseek(messages, api_key, max_tokens=1500) # GoT might need more tokens

        if llm_response and not llm_response.startswith("Error:"):
            processed_sample = process_got_response(problem, llm_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate valid GoT response for problem: {problem}. Error: {llm_response}")
            generated_samples.append({
                "problem": problem,
                "thought_exploration": "GENERATION_ERROR",
                "raw_response": llm_response or "No response"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} GoT samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Graph of Thoughts (GoT) dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/got_dataset.jsonl", help="Path to save the dataset.")
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

# สร้างข้อความสำหรับหมวดหมู่ graph_of_thoughts
categories = {
    "graph_of_thoughts": [
        "ออกแบบกลยุทธ์การตลาดสำหรับผลิตภัณฑ์ใหม่ที่ต้องการเจาะตลาดกลุ่มวัยรุ่น โดยพิจารณาช่องทางต่างๆ และผลกระทบที่เชื่อมโยงกัน",
        "วิเคราะห์สาเหตุและผลกระทบของปัญหาการเปลี่ยนแปลงสภาพภูมิอากาศ โดยแสดงความสัมพันธ์ของปัจจัยต่างๆ ในรูปแบบเครือข่าย",
        "วางแผนการพัฒนาเมืองอัจฉริยะ (Smart City) โดยคำนึงถึงความเชื่อมโยงของระบบขนส่ง พลังงาน และการจัดการทรัพยากร",
        "หากต้องการแก้ไขปัญหารถติดในกรุงเทพมหานครอย่างยั่งยืน ควรพิจารณาแนวทางใดบ้าง และแต่ละแนวทางส่งผลกระทบต่อกันอย่างไร",
        "ออกแบบระบบการจัดการภัยพิบัติสำหรับชุมชนชายฝั่งทะเล โดยเชื่อมโยงการเตือนภัย การอพยพ และการฟื้นฟู",
        "วิเคราะห์ปัจจัยที่ส่งผลต่อความสำเร็จของธุรกิจสตาร์ทอัพ โดยแสดงความสัมพันธ์ระหว่างทีมงาน ผลิตภัณฑ์ ตลาด และแหล่งเงินทุน",
        "วางแผนการวิจัยเพื่อค้นหาวิธีรักษาโรคใหม่ โดยพิจารณาแนวทางการทดลองต่างๆ และความเชื่อมโยงของผลลัพธ์ที่อาจเกิดขึ้น",
        "ออกแบบนโยบายสาธารณะเพื่อลดความเหลื่อมล้ำทางการศึกษา โดยพิจารณาปัจจัยทางเศรษฐกิจ สังคม และโครงสร้างพื้นฐานที่เกี่ยวข้อง",
        "วิเคราะห์ความสัมพันธ์ระหว่างปัจจัยต่างๆ ที่ส่งผลต่อการตัดสินใจลงทุนในตลาดหุ้น เช่น สภาพเศรษฐกิจโลก อัตราดอกเบี้ย และผลประกอบการบริษัท",
        "วางแผนการจัดการทรัพยากรน้ำในภาวะภัยแล้ง โดยคำนึงถึงความต้องการใช้น้ำของภาคส่วนต่างๆ และผลกระทบของการจัดสรรน้ำ"
    ]
}

if __name__ == "__main__":
    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_graph_of_thoughts.csv")

    rows = []
    for label, texts in categories.items():
        for text in texts:
            rows.append([str(uuid.uuid4()), text, label])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")

