import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ career_growth
categories = {
    "career_growth": [
        "อยากเลื่อนตำแหน่ง ต้องพัฒนาทักษะอะไรบ้าง",
        "ทักษะที่ตลาดงานต้องการมากที่สุดในปีนี้",
        "วิธีสร้าง Career Path ให้เติบโตในสายงาน",
        "การพัฒนา Soft Skills สำคัญต่อความก้าวหน้าอย่างไร",
        #...
        "ทักษะที่จำเป็นสำหรับการเป็นผู้จัดการ"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_career_growth.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_career_growth.csv")
