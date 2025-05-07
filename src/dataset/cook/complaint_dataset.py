import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ complaint
categories = {
    "complaint": [
        "อินเทอร์เน็ตบ้านล่มอีกแล้ว แจ้งไปหลายรอบก็ยังไม่แก้",
        "โทรหาคอลเซ็นเตอร์ไม่เคยติดเลย รอสายนานมาก",
        #...
        "ถูกรบกวนจาก SMS ขยะและแก๊งคอลเซ็นเตอร์",
        "คุณภาพสัญญาณทีวีดิจิทัลไม่ดี"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_complaint.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_complaint.csv")
