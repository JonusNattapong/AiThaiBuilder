import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ fan_culture
categories = {
    "fan_culture": [
        "วิธีซัพพอร์ตศิลปิน/ดาราที่ชอบอย่างสร้างสรรค์",
        "วัฒนธรรมแฟนด้อม (Fandom) ในวงการ K-Pop",
        #...
        "การเปลี่ยนแปลงของวัฒนธรรมแฟนคลับในยุคดิจิทัล",
        "พลังของแฟนคลับในการขับเคลื่อนประเด็นทางสังคม"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_fan_culture.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_fan_culture.csv")
