import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ emotion
categories = {
    "emotion": [
        "รู้สึกท้อแท้จังวันนี้ ทำอะไรก็ไม่สำเร็จ",
        "มีความสุขมากเลย ได้เจอเพื่อนเก่า",
        #...
        "สงสารเด็กกำพร้า",
        "อึดอัดที่ต้องใส่หน้ากากตลอดเวลา"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_emotion.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_emotion.csv")
