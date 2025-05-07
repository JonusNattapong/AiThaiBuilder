import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ science
categories = {
    "science": [
        "ทฤษฎีสัมพัทธภาพของไอน์สไตน์",
        "การค้นพบโครงสร้างดีเอ็นเอ",
        "วิทยาศาสตร์คอมพิวเตอร์และอัลกอริทึม",
        "การศึกษาเรื่องความเจ็บปวด"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_science.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_science.csv")
