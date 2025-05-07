import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ fiction
categories = {
    "fiction": [
        "หญิงสาวค้นพบประตูมิติที่นำเธอไปสู่อนาคต",
        "เรื่องราวของผีสาวที่ปรากฏตัวเฉพาะตอนบ่ายสองโมงตรง",
        #...
        "เรื่องราวของกลุ่มคนที่ได้รับพลังพิเศษหลังเกิดภัยพิบัติ",
        "บทสนทนาสุดท้ายก่อนโลกจะแตกสลาย"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_fiction.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_fiction.csv")
