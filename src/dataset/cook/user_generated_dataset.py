import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ user_generated
categories = {
    "user_generated_status": [
        "วันนี้อากาศดีจัง ☀️",
        "กำลังเดินทางไปทำงาน 🚗💨",
        # ...
    ],
    "user_generated_comment": [
        "เห็นด้วยเลยค่ะ 👍",
        "รูปสวยมากเลยค่ะ 😍",
        # ...
        "ฝันดีนะคะ 😴"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_user_generated_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_user_generated.csv")
