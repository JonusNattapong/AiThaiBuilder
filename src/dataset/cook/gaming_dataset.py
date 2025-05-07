import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ gaming
categories = {
    "gaming_news_reviews": [
        "รีวิวเกม AAA ใหม่ล่าสุด กราฟิกสวยอลังการ",
        "ข่าวหลุดเกมภาคต่อที่ทุกคนรอคอย",
        # ...
    ],
    "gaming_tips_tricks": [
        "วิธีผ่านด่านบอสสุดโหดในเกม...",
        "เทคนิคการฟาร์มของในเกม RPG ให้ได้ไว",
        # ...
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_gaming.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_gaming.csv")
