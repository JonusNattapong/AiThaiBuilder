import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ nonprofit
categories = {
    "nonprofit": [
        "มูลนิธิกระจกเงา ช่วยเหลือผู้ด้อยโอกาส",
        "โครงการอาสาสมัครเพื่อสังคม",
        "กิจกรรมรณรงค์ต่อต้านยาเสพติด",
        "องค์กรส่งเสริมศิลปวัฒนธรรมท้องถิ่น"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_nonprofit.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_nonprofit.csv")
