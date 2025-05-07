import csv
import uuid
import os

# Import data categories
from offensive_language_data import offensive_language_data
from adult_content_data import adult_content_data
from hate_speech_data import hate_speech_data
from misinformation_data import misinformation_data
from illegal_activities_data import illegal_activities_data
from dangerous_instructions_data import dangerous_instructions_data
from extreme_violence_data import extreme_violence_data
from self_harm_content_data import self_harm_content_data
from threatening_content_data import threatening_content_data
from inappropriate_qa_data import inappropriate_qa_data
from harmful_text_generation_data import harmful_text_generation_data
from scam_templates_data import scam_templates_data

# คำเตือน: สคริปต์นี้สร้างข้อมูลที่มีเนื้อหาอ่อนไหวซึ่งอาจไม่เหมาะสมกับทุกบริบท
# วัตถุประสงค์เพื่อการวิจัยและพัฒนาระบบกรองเนื้อหาอัตโนมัติเท่านั้น

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# รวมข้อมูลจากทุกหมวดหมู่
categories = {
    "offensive_language": offensive_language_data,
    "adult_content": adult_content_data,
    "hate_speech": hate_speech_data,
    "misinformation": misinformation_data,
    "illegal_activities": illegal_activities_data,
    "dangerous_instructions": dangerous_instructions_data,
    "extreme_violence": extreme_violence_data,
    "self_harm_content": self_harm_content_data,
    "threatening_content": threatening_content_data,
    "inappropriate_qa": inappropriate_qa_data,
    "harmful_text_generation": harmful_text_generation_data,
    "scam_templates": scam_templates_data
}

# สร้างรายการข้อมูลพร้อม ID และหมวดหมู่ย่อย
rows = []
for category, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, category])

# บันทึกเป็นไฟล์ CSV
output_file = os.path.join(output_dir, 'thai_uncensored_dataset.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'category'])
    writer.writerows(rows)

print(f"Created {output_file} - for research purposes only!")
print("คำเตือน: ข้อมูลดิบนี้มีเนื้อหาอ่อนไหวและไม่เหมาะสม สร้างขึ้นเพื่อวัตถุประสงค์ในการวิจัยเท่านั้น")
print("Warning: This raw dataset contains sensitive and inappropriate content, created for research purposes only.")
