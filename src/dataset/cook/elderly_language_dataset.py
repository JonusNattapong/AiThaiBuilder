import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ elderly_language
categories = {
    "elderly_language": [
        "วันนี้อากาศร้อนมากนะลูก คิดถึงตอนยังเป็นเด็ก วิ่งเล่นไม่กลัวแดดเลย",
        "สมัยก่อนไม่มีหรอกนะโทรศัพท์มือถือ อยากคุยก็ต้องเขียนจดหมาย",
        #...
        "จำความได้ว่าแถวนี้เคยเป็นทุ่งนา",
        "บุญคุณต้องทดแทนนะลูก ใครทำดีกับเราต้องจำไว้"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_elderly_language.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_elderly_language.csv")
