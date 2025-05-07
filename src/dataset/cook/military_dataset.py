import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ military
categories = {
    "military": [
        "กองทัพไทยกับการช่วยเหลือประชาชนในภัยพิบัติ",
        "การฝึกซ้อมรบร่วมระหว่างประเทศ",
        "กองทัพอากาศปฏิบัติภารกิจฝนหลวง"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_military.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_military.csv")
