import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ coding
categories = {
    "coding": [
        "วิธีเขียนโปรแกรม Python เบื้องต้นสำหรับมือใหม่",
        "การใช้งาน Git และ GitHub สำหรับควบคุมเวอร์ชัน",
        #...
        "หลักการของ Security ในการพัฒนาซอฟต์แวร์",
        "การเขียนโปรแกรมภาษา TypeScript เพิ่มความปลอดภัยให้ JavaScript",
        "วิธีสร้างเว็บไซต์ง่ายๆ ด้วย HTML และ CSS"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_coding.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_coding.csv")
