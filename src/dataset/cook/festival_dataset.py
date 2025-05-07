import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ festival
categories = {
    "festival": [
        "สงกรานต์ปีนี้เล่นน้ำที่ไหนดี",
        "ประเพณีลอยกระทง สืบสานวัฒนธรรมไทย",
        #...
        "เทศกาล La Tomatina ปามะเขือเทศที่สเปน",
        "ประเพณีผีตาโขนที่เลย"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_festival.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_festival.csv")
