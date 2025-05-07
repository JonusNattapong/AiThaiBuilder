import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ life_hack
categories = {
    "life_hack": [
        "วิธีเปิดฝาขวดโหลที่แน่นเกินไปง่ายๆ",
        "เทคนิคการพับเสื้อผ้าให้ประหยัดพื้นที่ในกระเป๋าเดินทาง",
        "วิธีทำความสะอาดคราบน้ำมันในครัว",
        "เทคนิคเลือกซื้ออะโวคาโดให้สุกกำลังดี"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_life_hack.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_life_hack.csv")
