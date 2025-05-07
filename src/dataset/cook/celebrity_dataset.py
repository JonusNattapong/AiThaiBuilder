import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ celebrity
categories = {
    "celebrity": [
        "ชื่นชมนักแสดงคนนี้มาก เล่นได้ทุกบทบาทเลย",
        "เน็ตไอดอลคนนี้เป็นแรงบันดาลใจในการลดน้ำหนัก",
        #...
        "ชอบฟัง Podcast ของคนดังคนนี้",
        "ชื่นชมในความสามารถพิเศษของดาราคนนี้"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_celebrity.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_celebrity.csv")
