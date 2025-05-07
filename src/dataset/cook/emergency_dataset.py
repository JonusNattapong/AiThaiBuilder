import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ emergency
categories = {
    "emergency": [
        "เกิดเหตุไฟไหม้ที่ตลาดเมื่อคืนนี้",
        "น้ำท่วมหนักหลายพื้นที่ ต้องการความช่วยเหลือด่วน",
        #...
        "เหตุการณ์ไฟฟ้าดับเป็นวงกว้าง",
        "การแจ้งเหตุฉุกเฉินอื่นๆ"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_emergency.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_emergency.csv")
