import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ learning
categories = {
    "learning": [
        "วิธีเรียนรู้ทักษะใหม่ๆ ให้มีประสิทธิภาพ",
        "เทคนิคการอ่านหนังสือให้จำได้นาน",
        "วิธีฝึกทักษะการปรับตัว (Adaptability)",
        "การเรียนรู้การใช้โปรแกรมคอมพิวเตอร์ต่างๆ"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_learning.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_learning.csv")
