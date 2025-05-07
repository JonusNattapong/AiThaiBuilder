import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ promotion
categories = {
    "promotion_general": [
        "ซื้อ 1 แถม 1 วันนี้วันเดียวเท่านั้น!",
        "ลดราคาสูงสุด 70% ทุกรายการ",
        # ...
    ],
    "promotion_food": [
        "สั่งชุดสุดคุ้ม ลดทันที 50 บาท",
        "บุฟเฟต์มา 4 จ่าย 3",
        # ...
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_promotion.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_promotion.csv")
