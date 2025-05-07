import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ happy
categories = {
    "happy": [
        "วันนี้มีความสุขมาก ได้เจอคนที่ชอบ",
        "ดีใจสุดๆ สอบผ่านแล้ว!",
        "ดีใจที่ได้เห็นรุ้งกินน้ำหลังฝนตก"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_happy_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_happy.csv")
