import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ automotive
categories = {
    "automotive": [
        "รีวิวรถยนต์ไฟฟ้ารุ่นใหม่ล่าสุด",
        "เทคโนโลยีรถยนต์ไร้คนขับ (Autonomous Driving)",
        #...
        "เทคโนโลยีการแสดงผลบนกระจกหน้า (Head-up Display)",
        "การเลือกซื้อจักรยานยนต์"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_automotive.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_automotive.csv")
