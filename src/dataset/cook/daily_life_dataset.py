import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ daily_life
categories = {
    "daily_life": [
        "ตื่นนอนตอนเช้าต้องดื่มกาแฟก่อนเลย",
        "วันนี้กินอะไรเป็นมื้อกลางวันดีนะ",
        #...
        "เช็คตารางงาน ตารางเรียน",
        "ดื่มน้ำเยอะๆ ระหว่างวัน"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/daily_life_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_daily_life.csv")
