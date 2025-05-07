import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ traditional_culture
categories = {
    "traditional_culture": [
        "ประเพณีสงกรานต์และการรดน้ำดำหัวผู้ใหญ่",
        "การละเล่นพื้นบ้านของไทย เช่น มอญซ่อนผ้า",
        "ขนมไทยโบราณและความหมายมงคล",
        "ประเพณีล้างเท้าขอขมาพ่อแม่"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_traditional_culture.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_traditional_culture.csv")
