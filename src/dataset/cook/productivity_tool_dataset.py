import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ productivity_tool
categories = {
    "productivity_tool_software": [
        "แนะนำโปรแกรมจัดการงาน (Task Management) เช่น Trello, Asana",
        "วิธีใช้ Google Calendar วางแผนตารางงานอย่างมีประสิทธิภาพ",
        # ...
    ],
    "productivity_tool_hardware": [
        "การเลือกใช้คีย์บอร์ด Ergonomic เพื่อลดอาการปวดข้อมือ",
        "ประโยชน์ของจอคอมพิวเตอร์แบบ Ultrawide สำหรับการทำงาน",
        # ...
    ],
    "productivity_tool": [
        "วิธีใช้ Notion จัดระเบียบชีวิตและการทำงาน",
        "รวมแอปพลิเคชันช่วยจัดการเวลาและ To-Do List",
        "แนะนำเครื่องมือสำหรับจดบันทึกและจัดการความรู้ (PKM)",
        "การใช้ Trello หรือ Asana ในการบริหารโปรเจกต์",
        "เทคนิคใช้ Google Calendar วางแผนตารางงานอย่างมีประสิทธิภาพ",
        "แอป Pomodoro Timer ช่วยเพิ่มสมาธิในการทำงาน",
        "การใช้ Evernote เก็บข้อมูลและไอเดียต่างๆ",
        "เครื่องมือช่วยสร้าง Mind Map สำหรับระดมสมองและวางแผน",
        "แนะนำแอป Habit Tracker ช่วยสร้างนิสัยที่ดี",
        "การใช้ Slack หรือ Microsoft Teams สื่อสารในทีม",
        "เครื่องมือช่วยจัดการรหัสผ่าน (Password Manager)",
        "แอป Focus Mode ช่วยลดสิ่งรบกวนจากมือถือ",
        "การใช้ Google Drive หรือ Dropbox จัดเก็บและแชร์ไฟล์",
        "เครื่องมือช่วยอ่านเร็วและจับใจความสำคัญ",
        "แนะนำแอปจดบันทึกเสียงและถอดความ",
        "การใช้ Zapier หรือ IFTTT สร้าง Workflow อัตโนมัติ",
        "เครื่องมือช่วยตรวจไวยากรณ์และสไตล์การเขียน (Grammarly)",
        "แอปสำหรับจัดการอีเมลให้เป็นระเบียบ",
        "การใช้เครื่องมือ Whiteboard ออนไลน์สำหรับการประชุม",
        "แนะนำแอปสำหรับอ่าน Feed ข่าวสาร (RSS Reader)",
        "เครื่องมือช่วยสร้าง Presentation ให้น่าสนใจ",
        "แอปสำหรับติดตามค่าใช้จ่ายและทำบัญชีส่วนตัว",
        "การใช้เครื่องมือ Screenshot และ Screen Recording",
        "แนะนำแอปสำหรับฝึกสมาธิและผ่อนคลาย (Meditation App)",
        "เครื่องมือช่วยออกแบบกราฟิกง่ายๆ (Canva)",
        "แอปสำหรับอ่าน E-book และจัดการหนังสือ",
        "การใช้เครื่องมือ VPN เพื่อความปลอดภัย",
        "แนะนำแอปสำหรับเรียนภาษาต่างประเทศ",
        "เครื่องมือช่วยสร้างและจัดการเว็บไซต์/บล็อก",
        "แอปสำหรับสแกนเอกสารเป็นไฟล์ดิจิทัล",
        "การใช้เครื่องมือ Video Conferencing (Zoom, Google Meet)",
        "แนะนำแอปสำหรับออกกำลังกายที่บ้าน",
        "เครื่องมือช่วยจัดการ Social Media",
        "แอปสำหรับจดบันทึกการประชุม (Meeting Notes)",
        "การใช้เครื่องมือสำรองข้อมูล (Backup Tool)",
        "แนะนำแอปสำหรับฟัง Podcast",
        "เครื่องมือช่วยสร้าง Resume และ Portfolio",
        "แอปสำหรับติดตามพัสดุ",
        "การใช้เครื่องมือ Developer Tools ในเว็บบราวเซอร์",
        "แนะนำแอปสำหรับวางแผนการเดินทาง",
        "เครื่องมือช่วยวิเคราะห์ข้อมูล (Data Analysis Tool)",
        "แอปสำหรับจัดการสูตรอาหารและวางแผนมื้ออาหาร",
        "การใช้เครื่องมือ Remote Desktop",
        "แนะนำแอปสำหรับแต่งรูปภาพบนมือถือ",
        "เครื่องมือช่วยสร้างแบบสำรวจออนไลน์ (Survey Tool)",
        "แอปสำหรับติดตามรอบเดือนของผู้หญิง",
        "การใช้เครื่องมือ Cloud Storage อื่นๆ (OneDrive, iCloud)",
        "แนะนำแอปสำหรับค้นหางาน",
        "เครื่องมือช่วยสร้าง QR Code",
        "แอปสำหรับควบคุมอุปกรณ์ Smart Home"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_productivity_tool.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_productivity_tool.csv")
