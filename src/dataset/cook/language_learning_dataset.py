import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ language_learning
categories = {
    "language_learning": [
        "วิธีเรียนภาษาอังกฤษให้เก่งเร็ว",
        "แนะนำแอปพลิเคชันสำหรับฝึกภาษา",
        "วิธีรักษาแรงจูงใจในการเรียนภาษา",
        "การใช้แอป Duolingo, Memrise, Babbel"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_language_learning.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_language_learning.csv")
