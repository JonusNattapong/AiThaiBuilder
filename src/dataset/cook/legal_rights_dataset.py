import csv
import uuid
import os

# สร้างข้อความสำหรับหมวดหมู่ legal_rights
categories = {
    "legal_rights": [
        "สิทธิของผู้บริโภคในการซื้อสินค้าและบริการ",
        "กฎหมายแรงงานที่ลูกจ้างควรรู้",
        "การถูกหลอกลวงในการทำธุรกรรมทางการเงิน",
        "กฎหมายเกี่ยวกับภาษีเงินได้บุคคลธรรมดา"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
# Ensure output directory exists
output_dir = os.path.join("..", "..", "DataOutput")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "thai_dataset_legal_rights.csv") # Corrected path and standardized filename

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print(f"Created {output_file}") # Updated print statement to reflect correct path
