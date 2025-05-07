import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ debate
categories = {
    "debate": [
        "ควรมีการเรียนออนไลน์แบบถาวรเป็นทางเลือกหรือไม่?",
        "การสวมหน้ากากอนามัยควรเป็นสิทธิส่วนบุคคลหรือข้อบังคับ?",
        #...
        "ควรมีการปฏิรูปกองทัพอย่างไร?",
        "การใช้เทคโนโลยีชีวภาพในการผลิตอาหาร (GMOs) ปลอดภัยหรือไม่?"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_debate.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_debate.csv")
