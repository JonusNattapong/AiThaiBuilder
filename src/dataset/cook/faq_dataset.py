import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ faq
categories = {
    "faq_general": [
        "สินค้าชิ้นนี้มีรับประกันหรือไม่",
        "สามารถคืนสินค้าได้ภายในกี่วัน",
        # ...
    ],
    "faq_technical": [
        "ทำไมฉันถึงเชื่อมต่อ Wi-Fi ไม่ได้",
        "วิธีรีเซ็ต пароль (password) ทำอย่างไร",
        # ...
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_faq.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_faq.csv")
