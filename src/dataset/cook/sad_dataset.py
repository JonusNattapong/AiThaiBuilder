import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ sad
categories = {
    "sad": [
        "รู้สึกเศร้าจังวันนี้ ไม่มีใครเข้าใจเลย",
        "อกหักครั้งนี้มันเจ็บปวดเหลือเกิน",
        "เศร้าใจที่ไม่สามารถแสดงความรู้สึกที่แท้จริงออกไปได้",
        "รู้สึกเหมือนโลกทั้งใบกำลังถล่มลงมา"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_sad_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_sad.csv")
