import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ coding
categories = {
    "coding": [
        "วิธีเขียนโปรแกรม Python เบื้องต้นสำหรับมือใหม่",
        "การใช้งาน Git และ GitHub สำหรับควบคุมเวอร์ชัน",
        "แนะนำภาษาโปรแกรมที่เหมาะกับการพัฒนาเว็บ",
        "หลักการเขียนโค้ดที่ดี (Clean Code)",
        "วิธี Debug โค้ดหาข้อผิดพลาด",
        "การใช้งาน Framework ยอดนิยม เช่น React, Vue, Angular",
        "ความแตกต่างระหว่าง Frontend และ Backend Development",
        "แนะนำ Text Editor และ IDE สำหรับเขียนโค้ด",
        "การเขียนโปรแกรมเชิงวัตถุ (OOP) คืออะไร",
        "วิธีสร้าง API ด้วย Node.js และ Express",
        "การจัดการ Database ด้วย SQL และ NoSQL",
        "แนะนำแหล่งเรียนรู้การเขียนโปรแกรมออนไลน์",
        "การเขียน Unit Test และ Integration Test",
        "วิธีใช้งาน Docker สำหรับสร้าง Container",
        "หลักการของ Agile และ Scrum ในการพัฒนาซอฟต์แวร์",
        "การเขียน Mobile App ด้วย React Native หรือ Flutter",
        "แนะนำ Extension VS Code ที่ช่วยให้เขียนโค้ดง่ายขึ้น",
        "การใช้งาน Command Line เบื้องต้นสำหรับโปรแกรมเมอร์",
        "วิธี Deploy เว็บไซต์ขึ้น Server",
        "หลักการออกแบบ UX/UI ที่ดี",
        "การเขียนโปรแกรมภาษา Java สำหรับ Enterprise Application",
        "วิธีจัดการ Dependency ในโปรเจกต์",
        "การใช้งาน Cloud Platform เช่น AWS, Google Cloud, Azure",
        "หลักการของ Data Structures และ Algorithms",
        "การเขียนโปรแกรมภาษา C# สำหรับพัฒนาเกมด้วย Unity",
        "วิธีสร้าง Chatbot ด้วย Python",
        "การทำ Web Scraping เพื่อดึงข้อมูลจากเว็บไซต์",
        "แนะนำ Community และ Forum สำหรับโปรแกรมเมอร์",
        "การเขียนโปรแกรมภาษา Go (Golang) สำหรับ Backend",
        "วิธีตั้งค่า Environment สำหรับพัฒนาโปรแกรม",
        "การใช้งาน CI/CD เพื่อทดสอบและ Deploy อัตโนมัติ",
        "หลักการของ RESTful API Design",
        "การเขียนโปรแกรมภาษา Swift สำหรับพัฒนา iOS App",
        "วิธีจัดการ Error Handling ในโปรแกรม",
        "การใช้งาน Regular Expression (Regex)",
        "แนะนำหนังสือเกี่ยวกับการเขียนโปรแกรมที่ควรอ่าน",
        "การเขียนโปรแกรมภาษา Kotlin สำหรับพัฒนา Android App",
        "วิธีสร้าง Portfolio สำหรับโปรแกรมเมอร์",
        "การใช้งาน GraphQL แทน REST API",
        "หลักการของ Functional Programming",
        "การเขียนโปรแกรมภาษา R สำหรับ Data Science",
        "วิธีเข้าร่วม Open Source Project",
        "การใช้งาน Package Manager เช่น npm, pip, composer",
        "หลักการของ Microservices Architecture",
        "การเขียนโปรแกรมภาษา PHP สำหรับพัฒนาเว็บ",
        "วิธีเตรียมตัวสัมภาษณ์งานตำแหน่งโปรแกรมเมอร์",
        "การใช้งาน Version Control อื่นๆ เช่น Mercurial, SVN",
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
