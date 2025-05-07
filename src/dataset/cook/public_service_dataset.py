import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ public_service
categories = {
    "public_service": [
        "ขั้นตอนการทำบัตรประชาชนใหม่",
        "ติดต่อหน่วยงานราชการต้องเตรียมเอกสารอะไรบ้าง",
        "แจ้งปัญหาน้ำประปาไม่ไหล ไฟฟ้าดับ",
        "การยื่นภาษีออนไลน์ผ่านเว็บไซต์กรมสรรพากร",
        "การทำใบขับขี่ที่กรมการขนส่งทางบก",
        #...
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_public_service.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_public_service.csv")
