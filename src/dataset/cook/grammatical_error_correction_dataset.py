"""
Module สำหรับสร้างชุดข้อมูล Grammatical Error Correction ในภาษาไทย
ใช้สำหรับการเรียนรู้การแก้ไขข้อผิดพลาดทางไวยากรณ์
"""

# ข้อมูลตัวอย่างประโยคที่มีข้อผิดพลาดทางไวยากรณ์และการแก้ไข
grammatical_error_examples = {
    'word_order': [
        "ฉันชอบมากๆ อาหารร้านนี้", # ผิด
        "ฉันชอบอาหารร้านนี้มากๆ"   # ถูก
    ],
    'classifier_usage': [
        "ฉันซื้อหนังสือสองเล่ม", # ถูก
        "ฉันซื้อหนังสือสอง"      # ผิด
    ],
    'verb_form': [
        "เขากำลังวิ่งเมื่อวาน",  # ผิด
        "เขาวิ่งเมื่อวาน"        # ถูก
    ],
    'missing_subject': [
        "ไปโรงเรียนทุกวัน",     # ผิด
        "ฉันไปโรงเรียนทุกวัน"   # ถูก
    ],
    'wrong_preposition': [
        "เขาอยู่บนรถไฟ",       # ผิด
        "เขาอยู่ในรถไฟ"        # ถูก
    ],
    'tense_inconsistency': [
        "เมื่อวานฉันกำลังไปตลาดและซื้อผลไม้",  # ผิด
        "เมื่อวานฉันไปตลาดและซื้อผลไม้"       # ถูก
    ],
    'redundant_words': [
        "เขาพูดบอกว่าจะมาสาย", # ผิด
        "เขาบอกว่าจะมาสาย"     # ถูก
    ],
    'wrong_conjunction': [
        "ฉันชอบกินข้าวกับดูหนัง", # ผิด
        "ฉันชอบกินข้าวและดูหนัง"  # ถูก
    ],
    'pronoun_agreement': [
        "ทุกคนต้องทำงานของเขาให้เสร็จ", # ผิด
        "ทุกคนต้องทำงานของตนให้เสร็จ"   # ถูก
    ],
    'formal_level_inconsistency': [
        "กรุณารอสักครู่นะ ฉันจะรีบมา", # ผิด (ผสมระหว่างทางการกับไม่ทางการ)
        "กรุณารอสักครู่นะคะ ดิฉันจะรีบมา" # ถูก
    ]
}

# ตัวอย่างประโยคที่ใช้เพื่อให้โมเดลสร้างคู่ประโยคผิด-ถูก
error_generation_prompts = [
    "เขาทำงานหนักมากเพื่อสอบให้ผ่าน",
    "วันนี้อากาศดีมาก ท้องฟ้าสีฟ้าสดใส",
    "นักเรียนทุกคนต้องส่งการบ้านภายในวันศุกร์",
    "อาหารไทยมีรสชาติเผ็ดและหลากหลาย",
    "ฉันต้องการซื้อเสื้อผ้าใหม่สำหรับงานแต่งงาน",
    "คุณแม่กำลังทำอาหารเย็นในครัว",
    "คอมพิวเตอร์เครื่องนี้ทำงานได้เร็วมาก",
    "ประเทศไทยมีสถานที่ท่องเที่ยวสวยงามมากมาย",
    "นักวิทยาศาสตร์ค้นพบยารักษาโรคชนิดใหม่",
    "เด็กๆ กำลังเล่นฟุตบอลในสนามหญ้า"
]

# ประเภทของข้อผิดพลาดทางไวยากรณ์ที่ต้องการให้โมเดลสร้าง
error_types = {
    'word_order': 'การเรียงลำดับคำผิด เช่น "ฉันชอบมากๆ อาหารร้านนี้" แทนที่จะเป็น "ฉันชอบอาหารร้านนี้มากๆ"',
    'classifier_usage': 'การใช้ลักษณนามผิด หรือไม่ใส่ลักษณนาม เช่น "ฉันซื้อหนังสือสอง" แทนที่จะเป็น "ฉันซื้อหนังสือสองเล่ม"',
    'verb_form': 'การใช้รูปกริยาผิด เช่น "เขากำลังวิ่งเมื่อวาน" แทนที่จะเป็น "เขาวิ่งเมื่อวาน"',
    'missing_subject': 'การละประธาน เช่น "ไปโรงเรียนทุกวัน" แทนที่จะเป็น "ฉันไปโรงเรียนทุกวัน"',
    'wrong_preposition': 'การใช้คำบุพบทผิด เช่น "เขาอยู่บนรถไฟ" แทนที่จะเป็น "เขาอยู่ในรถไฟ"',
    'tense_inconsistency': 'การใช้กาลที่ไม่สอดคล้องกัน เช่น "เมื่อวานฉันกำลังไปตลาดและซื้อผลไม้" แทนที่จะเป็น "เมื่อวานฉันไปตลาดและซื้อผลไม้"',
    'redundant_words': 'การใช้คำฟุ่มเฟือย เช่น "เขาพูดบอกว่าจะมาสาย" แทนที่จะเป็น "เขาบอกว่าจะมาสาย"',
    'wrong_conjunction': 'การใช้คำเชื่อมผิด เช่น "ฉันชอบกินข้าวกับดูหนัง" แทนที่จะเป็น "ฉันชอบกินข้าวและดูหนัง"',
    'pronoun_agreement': 'การใช้คำสรรพนามไม่สอดคล้องกัน เช่น "ทุกคนต้องทำงานของเขาให้เสร็จ" แทนที่จะเป็น "ทุกคนต้องทำงานของตนให้เสร็จ"',
    'formal_level_inconsistency': 'ความไม่สอดคล้องของระดับภาษา เช่น "กรุณารอสักครู่นะ ฉันจะรีบมา" แทนที่จะเป็น "กรุณารอสักครู่นะคะ ดิฉันจะรีบมา"'
}

def process_grammatical_correction_response(original_text: str, llm_response: str):
    """
    ประมวลผลการตอบกลับจาก LLM สำหรับข้อมูล Grammatical Error Correction
    
    Args:
        original_text: ข้อความต้นฉบับที่ส่งให้ LLM
        llm_response: การตอบกลับจาก LLM
        
    Returns:
        Dictionary ที่มี original_text, error_text, corrected_text และ explanation
    """
    try:
        # พยายามแยกส่วนออกจากการตอบกลับ
        error_marker = "ประโยคที่มีข้อผิดพลาด:"
        correct_marker = "ประโยคที่ถูกต้อง:"
        explanation_marker = "คำอธิบาย:"
        
        # แยกส่วนประโยคที่มีข้อผิดพลาด
        error_start = llm_response.find(error_marker)
        correct_start = llm_response.find(correct_marker)
        explanation_start = llm_response.find(explanation_marker)
        
        error_text = ""
        corrected_text = ""
        explanation = ""
        
        # ดึงข้อมูลส่วนต่างๆ จากการตอบกลับ
        if error_start != -1 and correct_start != -1:
            error_text = llm_response[error_start + len(error_marker):correct_start].strip()
            
            if explanation_start != -1:
                corrected_text = llm_response[correct_start + len(correct_marker):explanation_start].strip()
                explanation = llm_response[explanation_start + len(explanation_marker):].strip()
            else:
                corrected_text = llm_response[correct_start + len(correct_marker):].strip()
        
        # ถ้าไม่พบรูปแบบตามที่คาดหวัง ลองใช้วิธีอื่น
        if not error_text or not corrected_text:
            lines = llm_response.split('\n')
            for i, line in enumerate(lines):
                if error_marker in line and i+1 < len(lines):
                    error_text = lines[i+1].strip()
                if correct_marker in line and i+1 < len(lines):
                    corrected_text = lines[i+1].strip()
                if explanation_marker in line and i+1 < len(lines):
                    explanation = lines[i+1].strip()
        
        return {
            "original_prompt": original_text,
            "error_text": error_text,
            "corrected_text": corrected_text,
            "explanation": explanation,
            "raw_response": llm_response
        }
    
    except Exception as e:
        print(f"Error processing grammatical correction response: {e}")
        return {
            "original_prompt": original_text,
            "error_text": "Processing Error",
            "corrected_text": "Processing Error",
            "explanation": f"Error: {str(e)}",
            "raw_response": llm_response
        }