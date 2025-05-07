import gradio as gr
from gradio.themes import Soft
import os
import sys
import json
import datetime

# Add utils directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key
from utils.deepseek_translation_utils import (
    generate_en_to_th_translation,
    generate_zh_to_th_translation
)

# Load configuration from config.json
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Load configurations
ALL_TASKS = config['ALL_TASKS']
DEFAULT_SYSTEM_PROMPT = config['DEFAULT_SYSTEM_PROMPT']
TASK_CONFIG = config['TASK_CONFIG']

# Map translation function strings to actual functions
TRANSLATION_FUNCTIONS = {
    'generate_en_to_th_translation': generate_en_to_th_translation,
    'generate_zh_to_th_translation': generate_zh_to_th_translation
}

# Update task configs to map function strings to actual functions
for task_name, task_info in TASK_CONFIG.items():
    if isinstance(task_info, dict) and 'translation_function' in task_info:
        func_name = task_info['translation_function']
        if isinstance(func_name, str) and func_name in TRANSLATION_FUNCTIONS:
            task_info['translation_function'] = TRANSLATION_FUNCTIONS[func_name]
    


def generate_dataset(
    api_key_input, system_prompt_custom, selected_task, num_samples, additional_instructions,
    *task_specific_inputs # This will capture all dynamic inputs
):
    api_key = get_deepseek_api_key(api_key_input)
    if not api_key or "Error: Deepseek API key not provided." in api_key:
         return "ข้อผิดพลาด: ไม่ได้ระบุ Deepseek API Key หรือ API Key ไม่ถูกต้อง", None, ""

    if not selected_task:
        return "ข้อผิดพลาด: กรุณาเลือก Task ที่ต้องการ", None, ""

    task_info = TASK_CONFIG.get(selected_task)
    
    # Map task_specific_inputs to named inputs based on task_info
    input_values = {}
    if task_info and "inputs" in task_info:
        input_keys = list(task_info["inputs"].keys())
        for i, key in enumerate(input_keys):
            if i < len(task_specific_inputs):
                input_values[key] = task_specific_inputs[i]
            else: # Should not happen if UI is synced with TASK_CONFIG
                input_values[key] = "" 
    else: # Generic task or task not in TASK_CONFIG
        # Assume the first dynamic input is the 'main_input'
        if task_specific_inputs:
            input_values["main_input"] = task_specific_inputs[0]
        else:
            input_values["main_input"] = "" # Default if no inputs somehow
        if len(task_specific_inputs) > 1:
            input_values["context_input"] = task_specific_inputs[1] # Generic context

    # Determine system prompt
    system_prompt = system_prompt_custom if system_prompt_custom.strip() else \
                    (task_info.get("system_prompt_default", DEFAULT_SYSTEM_PROMPT) if task_info else DEFAULT_SYSTEM_PROMPT)

    results = []
    raw_outputs_for_display = []

    for i in range(int(num_samples)): # Ensure num_samples is int
        # Construct user prompt
        if task_info:
            user_prompt = task_info["user_prompt_template"].format(
                **input_values, # Pass all collected input values
                additional_instructions=additional_instructions if additional_instructions else ""
            ).strip()
        else: # Generic fallback for tasks not in TASK_CONFIG
            user_prompt = f"สำหรับงาน '{selected_task}', สร้างข้อมูลที่เกี่ยวข้องกับ: {input_values.get('main_input', '')}\n{additional_instructions if additional_instructions else ''}".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add a progress update to the UI (optional, can be slow for many samples)
        # yield f"กำลังสร้างตัวอย่างที่ {i+1}/{num_samples}...\n\n" + "\n".join(raw_outputs_for_display), None
        
        # Use translation function if available
        if task_info and hasattr(task_info, 'get') and task_info.get('translation_function'):
            generated_text = task_info['translation_function'](input_values['main_input'], api_key, additional_instructions)
        else:
            generated_text = generate_with_deepseek(messages, api_key)

        if generated_text is None or "Error:" in generated_text:
            raw_outputs_for_display.append(f"ตัวอย่างที่ {i+1}: เกิดข้อผิดพลาด - {generated_text}")
            # Optionally skip adding to results or add error marker
            if task_info and task_info.get("output_format") == "jsonl":
                error_entry = {key: input_values.get(key, "") for key in task_info.get("jsonl_fields", [])[:-1]}
                error_entry[task_info.get("jsonl_fields", ["generated_answer"])[-1]] = f"GENERATION_ERROR: {generated_text}"
                results.append(json.dumps(error_entry, ensure_ascii=False))
            else:
                results.append(f"GENERATION_ERROR: {generated_text}")
            continue

        raw_outputs_for_display.append(f"ตัวอย่างที่ {i+1}:\n{generated_text}\n---")

        if task_info and task_info.get("output_format") == "jsonl":
            # Create JSONL entry
            entry = {}
            # Populate with inputs used for generation
            jsonl_fields_config = task_info.get("jsonl_fields", [])
            for field_key in jsonl_fields_config:
                if field_key == "generated_answer": # Or a more generic "generated_output"
                    entry[field_key] = generated_text
                elif field_key in input_values:
                    entry[field_key] = input_values[field_key]
                else: # Handle cases where a field in jsonl_fields might not be a direct input
                    entry[field_key] = None # Or some default
            results.append(json.dumps(entry, ensure_ascii=False))
        else:
            results.append(generated_text)

    output_text_display = f"สร้างข้อมูลเสร็จสิ้น {int(num_samples)} ตัวอย่างสำหรับ Task: {selected_task}\n\n"
    output_text_display += "\n".join(raw_outputs_for_display)
    
    # Create file content
    file_content = "\n".join(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_ext = 'jsonl' if task_info and task_info.get('output_format') == 'jsonl' else 'txt'
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_filename = os.path.join(data_dir, f"dataset_{selected_task.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.{output_filename_ext}")
    
    # Save to a temporary file for download
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(file_content)
        return output_text_display, gr.File(value=output_filename, visible=True), f"ไฟล์ที่สร้าง: {output_filename}"
    except Exception as e:
        return f"{output_text_display}\n\nเกิดข้อผิดพลาดในการบันทึกไฟล์: {e}", None, ""


def update_task_inputs(selected_task):
    task_info = TASK_CONFIG.get(selected_task)
    updates = []
    
    # Max number of dynamic inputs we might need (e.g., for QA: context + question)
    # This should match the number of gr.Textbox components defined as dynamic_inputs_components
    max_dynamic_inputs = 2 
    
    if task_info and "inputs" in task_info:
        idx = 0
        for name, config in task_info["inputs"].items():
            if idx < max_dynamic_inputs:
                updates.append(gr.Textbox(label=config["label"], lines=config.get("lines", 3), visible=True, interactive=True, value="")) # Clear previous value
                idx +=1
        # Hide remaining dynamic input slots
        for i in range(idx, max_dynamic_inputs):
            updates.append(gr.Textbox(visible=False, interactive=False, value=""))
        
        # Update system prompt placeholder
        system_prompt_placeholder = task_info.get("system_prompt_default", DEFAULT_SYSTEM_PROMPT)
        updates.append(gr.Textbox(placeholder=system_prompt_placeholder)) # This updates the placeholder of system_prompt_custom
        # Update task description
        updates.append(gr.Markdown(value=f"**รายละเอียด Task:** {task_info.get('description', 'ไม่มีคำอธิบาย')}"))

    else: # Generic or unconfigured task
        updates.append(gr.Textbox(label="ป้อนข้อมูลหลักสำหรับ Task:", lines=3, visible=True, interactive=True, value=""))
        # Hide other dynamic input slots
        for i in range(1, max_dynamic_inputs):
            updates.append(gr.Textbox(visible=False, interactive=False, value=""))
        
        updates.append(gr.Textbox(placeholder=DEFAULT_SYSTEM_PROMPT))
        description = f"**Task:** {selected_task} (การตั้งค่าทั่วไป)"
        if selected_task and selected_task not in TASK_CONFIG:
            description += "\n\n*หมายเหตุ: Task นี้ยังไม่ได้ตั้งค่าเฉพาะเจาะจง โปรดปรับ System Prompt และ Additional Instructions ตามความเหมาะสม หรือแก้ไข TASK_CONFIG ใน app.py*"
        updates.append(gr.Markdown(value=description))

    return updates


with gr.Blocks(theme=Soft()) as demo:
    gr.Markdown("#  générateur de jeux de données thaïlandais (Thai Dataset Generator)")
    gr.Markdown("เครื่องมือสร้างชุดข้อมูลภาษาไทยสำหรับ Fine-tuning โมเดล AI โดยใช้ Deepseek API")

    with gr.Row():
        api_key_input = gr.Textbox(label="Deepseek API Key", type="password", placeholder="ใส่ API Key ของคุณที่นี่ (sk-...) หรือตั้งค่าใน .env")
    
    with gr.Accordion("การตั้งค่าขั้นสูง (Advanced Settings)", open=False):
        system_prompt_custom = gr.Textbox(
            label="System Prompt (ไม่บังคับ)", 
            placeholder=DEFAULT_SYSTEM_PROMPT, 
            lines=3,
            info="ปรับแต่ง System Prompt เพื่อควบคุมลักษณะการตอบสนองของ AI (ถ้าเว้นว่างจะใช้ค่าเริ่มต้นของ Task)"
        )

    task_description_md = gr.Markdown("เลือก Task เพื่อดูรายละเอียด")

    selected_task_dd = gr.Dropdown(
        label="เลือก Task (Select Task)", 
        choices=ALL_TASKS,
        value="Text Generation" # Default task
    )
    
    with gr.Group() as dynamic_inputs_group:
        task_input_1 = gr.Textbox(label="Input 1", lines=3, visible=False, interactive=False)
        task_input_2 = gr.Textbox(label="Input 2", lines=7, visible=False, interactive=False)
    
    dynamic_inputs_components = [task_input_1, task_input_2] 

    additional_instructions_input = gr.Textbox(
        label="คำสั่งเพิ่มเติม (Additional Instructions)", 
        lines=2, 
        placeholder="เช่น: 'เน้นความเป็นทางการ', 'ใช้คำศัพท์ง่ายๆ', 'สร้าง 3 ย่อหน้า'"
    )
    num_samples_input = gr.Number(label="จำนวนตัวอย่าง (Number of Samples)", value=3, minimum=1, step=1, precision=0)

    generate_button = gr.Button("สร้างชุดข้อมูล (Generate Dataset)", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("## ผลลัพธ์ (Output)")
    
    output_status_text = gr.Textbox(label="สถานะการสร้าง / แสดงตัวอย่างผลลัพธ์", lines=10, interactive=False)
    download_file_component = gr.File(label="ดาวน์โหลดชุดข้อมูล (Download Dataset)", interactive=False, visible=False)
    generated_file_info = gr.Markdown("")


    # Link dropdown change to update dynamic inputs and system prompt placeholder
    # The outputs of update_task_inputs must match the components in dynamic_inputs_components
    # plus the system_prompt_custom (for placeholder) and task_description_md
    selected_task_dd.change(
        fn=update_task_inputs,
        inputs=[selected_task_dd],
        outputs=dynamic_inputs_components + [system_prompt_custom, task_description_md]
    )

    # Trigger initial update for default task
    demo.load(
        fn=update_task_inputs,
        inputs=[selected_task_dd], 
        outputs=dynamic_inputs_components + [system_prompt_custom, task_description_md]
    )
    
    generate_button.click(
        fn=generate_dataset,
        inputs=[
            api_key_input, system_prompt_custom, selected_task_dd, num_samples_input, additional_instructions_input
        ] + dynamic_inputs_components, 
        outputs=[output_status_text, download_file_component, generated_file_info]
    )

if __name__ == "__main__":
    demo.launch(share=False)
