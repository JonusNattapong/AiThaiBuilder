import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ job_seeking
categories = {
    "job_seeking": [
        "วิธีเขียนเรซูเม่ (Resume) อย่างไรให้โดดเด่นน่าสนใจ",
        "เทคนิคการเตรียมตัวสัมภาษณ์งานให้ได้งาน",
        "คำถามสัมภาษณ์งานที่พบบ่อยพร้อมแนวคำตอบ",
        "วิธีหางานจากเว็บไซต์และแพลตฟอร์มออนไลน์",
        #...
        "การเตรียมตัวสำหรับสัมภาษณ์งานเป็นภาษาอังกฤษ"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_job_seeking.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_job_seeking.csv")
