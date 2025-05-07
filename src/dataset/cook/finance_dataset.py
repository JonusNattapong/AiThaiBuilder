import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ health, finance, education
categories = {
    "finance": [
        "การวางแผนการเงินช่วยให้ชีวิตมั่นคง",
        "ควรเก็บเงินออมอย่างน้อย 10% ของรายได้",
        #...
        "การวางแผนเกษียณควรเริ่มเร็วที่สุด",
        "ควรหลีกเลี่ยงหนี้ที่ไม่จำเป็น"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_health_finance_education.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)