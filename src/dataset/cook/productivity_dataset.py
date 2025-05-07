import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ productivity
categories = {
    "productivity": [
        "เทคนิค Pomodoro ช่วยให้ทำงานมีสมาธิมากขึ้น",
        "วิธีจัดลำดับความสำคัญของงานด้วย Eisenhower Matrix",
        #...
        "การฝึกสติ (Mindfulness) เพื่อเพิ่มสมาธิ",
        "วิธีจัดการกับข้อมูลจำนวนมาก (Information Overload)"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_productivity.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_productivity.csv")
