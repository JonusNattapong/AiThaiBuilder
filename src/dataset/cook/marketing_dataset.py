import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ marketing
categories = {
    "marketing": [
        "กลยุทธ์การตลาดดิจิทัล (Digital Marketing)",
        "การทำ SEO ให้เว็บไซต์ติดอันดับ Google",
        "การตลาดผ่านโซเชียลมีเดีย (Social Media Marketing)",
        "Content Marketing คืออะไร ทำไมถึงสำคัญ",
        "การใช้ Influencer Marketing ให้ได้ผล",
        "การทำ Email Marketing อย่างมีประสิทธิภาพ",
        "การวิเคราะห์ข้อมูลลูกค้า (Customer Analytics)",
        "การสร้างแบรนด์ (Branding) ให้เป็นที่จดจำ",
        "การตลาดแบบบอกต่อ (Word-of-Mouth Marketing)",
        "การทำโฆษณาออนไลน์ (Online Advertising) เช่น Google Ads, Facebook Ads",
        "การแบ่งส่วนตลาด (Market Segmentation)",
        "การกำหนดกลุ่มเป้าหมาย (Targeting)",
        "การวางตำแหน่งผลิตภัณฑ์ (Positioning)",
        "ส่วนประสมทางการตลาด (Marketing Mix - 4Ps, 7Ps)",
        "การวิจัยตลาด (Market Research)",
        "การตลาดเชิงสัมพันธ์ (Relationship Marketing)",
        "การใช้ CRM (Customer Relationship Management)",
        "การตลาดอัตโนมัติ (Marketing Automation)",
        "การวัดผลแคมเปญการตลาด (Marketing ROI)",
        "การตลาดผ่านวิดีโอ (Video Marketing)",
        "การทำ Affiliate Marketing",
        "กลยุทธ์การตั้งราคา (Pricing Strategy)",
        "การสื่อสารการตลาดแบบบูรณาการ (IMC)",
        "การตลาดสำหรับธุรกิจ B2B และ B2C",
        "การสร้างประสบการณ์ลูกค้า (Customer Experience)",
        "การตลาดบนมือถือ (Mobile Marketing)",
        "การใช้ Data Visualization ในการนำเสนอข้อมูลการตลาด",
        "การตลาดแบบกองโจร (Guerrilla Marketing)",
        "การทำประชาสัมพันธ์ (Public Relations - PR)",
        "การตลาดเฉพาะบุคคล (Personalized Marketing)",
        "การวิเคราะห์คู่แข่ง (Competitor Analysis)",
        "การตลาดผ่าน Search Engine Marketing (SEM)",
        "การสร้าง Customer Journey Map",
        "การตลาดผ่าน Podcast",
        "การใช้ Chatbot ในการบริการลูกค้าและการตลาด",
        "การตลาดแบบยั่งยืน (Sustainable Marketing)",
        "การจัดอีเวนต์และการตลาดเชิงกิจกรรม (Event Marketing)",
        "การตลาดสำหรับสินค้าฟุ่มเฟือย (Luxury Marketing)",
        "การสร้างความภักดีของลูกค้า (Customer Loyalty)",
        "การตลาดระหว่างประเทศ (International Marketing)",
        "การใช้ Gamification ในการตลาด",
        "การตลาดแบบ Omni-channel",
        "การวิเคราะห์ SWOT Analysis",
        "การตลาดผ่าน LINE Official Account",
        "การทำ Neuromarketing",
        "การตลาดสำหรับธุรกิจบริการ",
        "การใช้ User-Generated Content (UGC)",
        "การตลาดในยุค AI",
        "จริยธรรมทางการตลาด"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_marketing.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_marketing.csv")
