import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ advice
categories = {
    "advice": [
        "ลองหาวิธีใหม่ๆ ดูสิ มันอาจจะช่วยแก้ปัญหานี้ได้นะ",
        "ไม่ต้องกังวลมากไปหรอก แค่พยายามอีกนิด เดี๋ยวก็ดีขึ้นเอง",
        #...
        "ลองมองโลกในแง่บวกดูบ้างนะ",
        "จำไว้ว่าพรุ่งนี้ก็เป็นวันใหม่แล้ว"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_advice.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_advice.csv")
