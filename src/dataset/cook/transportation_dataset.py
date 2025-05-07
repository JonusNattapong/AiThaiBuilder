import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ transportation
categories = {
    "transportation": [
        "การเดินทางด้วยรถไฟฟ้า BTS สะดวกและรวดเร็ว",
        "วิธีจองตั๋วเครื่องบินราคาถูก",
        "การวางแผนการเดินทางในช่วงวันหยุดยาว",
        "ความสะดวกสบายของห้องน้ำบนรถทัวร์"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_transportation.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_transportation.csv")
