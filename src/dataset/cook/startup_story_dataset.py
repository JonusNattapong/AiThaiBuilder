import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ startup_story
categories = {
    "startup_story": [
        "เรื่องราวการก่อตั้งบริษัทจากห้องเช่าเล็กๆ",
        "ความล้มเหลวครั้งแรกที่สอนบทเรียนสำคัญ",
        "วันที่เราเฉลิมฉลองความสำเร็จเล็กๆ ร่วมกัน",
        "เบื้องหลังการตัดสินใจที่ยากลำบากของผู้บริหาร"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_startup_story.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_startup_story.csv")
