import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ disaster
categories = {
    "disaster": [
        "วิธีรับมือเมื่อเกิดน้ำท่วมบ้าน ควรทำอย่างไร",
        "การเตรียมตัวหนีไฟป่าและป้องกันหมอกควัน",
        #...
        "การให้ความช่วยเหลือเพื่อนบ้านในยามเกิดภัย",
        "บทเรียนจากภัยพิบัติในอดีตและการปรับตัว"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_disaster.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_disaster.csv")
