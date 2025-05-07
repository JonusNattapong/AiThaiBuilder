import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ sports, lifestyle, job
categories = {
    "lifestyle": [
        "การจัดบ้านสไตล์มินิมอล",
        "เทรนด์แฟชั่นล่าสุดสำหรับวัยทำงาน",
        "การทำกระเป๋าเป็นงานอดิเรกที่สร้างสรรค์",
        "การจัดห้องครัวให้น่าใช้งาน"
    ]
}
    # สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_sports_lifestyle_job.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)