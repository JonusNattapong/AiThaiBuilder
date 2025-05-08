<p align="center">
  <img src="assets/banner.png" alt="AiThaiBuilder Banner" hight="200" width="200"/>
</p>

<h1 align="center">AiThaiBuilder</h1>
<p align="center"></p>

<p align="center">
  <a href="https://github.com/JonusNattapong/AiThaiBuilder/blob/main/LICENSE"><img src="https://img.shields.io/github/license/JonusNattapong/AiThaiBuilder?color=blue" alt="License"></a>
  <a href="https://github.com/JonusNattapong/AiThaiBuilder/stargazers"><img src="https://img.shields.io/github/stars/JonusNattapong/AiThaiBuilder?color=yellow" alt="Stars"></a>
  <a href="https://github.com/JonusNattapong/AiThaiBuilder/network/members"><img src="https://img.shields.io/github/forks/JonusNattapong/AiThaiBuilder?color=green" alt="Forks"></a>
  <a href="https://github.com/JonusNattapong/AiThaiBuilder/issues"><img src="https://img.shields.io/github/issues/JonusNattapong/AiThaiBuilder?color=red" alt="Issues"></a>
</p>

<p align="center">
  <b>AiThaiBuilder</b>
  โปรเจ็คที่ใช้สร้าง Ai เริ่มตั้งแต่ เตรียมข้อมูล เทรนโมเดล Upload and Save! โดย Focus ที่ Model ภาษาไทย เป็นหลัก รวมหลักการวิธีการ การเช็คคุณภาพ Datasest การทอสอบ Dataset และ อื่นๆสำหรับ การ Build Ai Thai
</p>

## 📚 สารบัญ

- [✨ คุณลักษณะเด่น](#-คุณลักษณะเด่น)
- [🛠️ การติดตั้ง](#️-การติดตั้ง)
- [🚀 วิธีการใช้งาน](#-วิธีการใช้งาน)
- [🧑‍💻 ทีมพัฒนา](#-ทีมพัฒนา)
- [🤝 การมีส่วนร่วม](#-การมีส่วนร่วม)
- [📄 ลิขสิทธิ์](#-ลิขสิทธิ์)

## ✨ คุณลักษณะเด่น

<table>
  <tr>
    <td width="50%">
      <h3>🔄 Dataset Generation แบบหลากหลาย</h3>
      <ul>
        <li>Text Generation (ทุกประเภทของข้อความ)</li>
        <li>Summarization (สรุปความ)</li>
        <li>Question Answering (ถาม-ตอบ)</li>
        <li>Translation (EN-TH, ZH-TH)</li>
        <li>Sentiment Analysis (วิเคราะห์ความรู้สึก)</li>
        <li>NER (Named Entity Recognition)</li>
        <li>Instruction Following (ทำตามคำสั่ง)</li>
        <li>Coreference Resolution (แก้ไขการอ้างถึง)</li>
        <li>Semantic Role Labeling (ติดป้ายบทบาททางความหมาย)</li>
        <li>ภาษาถิ่นไทยทั้ง 4 ภูมิภาค</li>
        <li>และอื่นๆ อีกมากมาย (ดูเพิ่มเติมได้ที่ <code>config/config.json</code>)</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🤖 Powered by Deepseek API</h3>
      <p>ใช้ language models ที่ทันสมัยที่สุดเพื่อสร้างข้อความภาษาไทยที่มีความเกี่ยวข้องกับบริบทและมีความสอดคล้องกัน พร้อมการควบคุมคุณภาพที่เข้มงวด</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>💬 Customizable Prompts</h3>
      <p>ผู้ใช้สามารถกำหนด system prompts และคำแนะนำเพิ่มเติมเพื่อปรับแต่งกระบวนการสร้างได้ตามต้องการ ช่วยให้ข้อมูลที่สร้างมีความเฉพาะเจาะจงตามความต้องการของแต่ละงาน</p>
    </td>
    <td width="50%">
      <h3>📊 Dataset Quality Analysis</h3>
      <p>มีเครื่องมือสำหรับวิเคราะห์และตรวจสอบคุณภาพของชุดข้อมูลพร้อมรายงาน PDF และการแสดงผลกราฟที่ครอบคลุมในโฟลเดอร์ <code>analysis_output</code></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📚 หลากหลายประเภทข้อมูล</h3>
      <p>รองรับการสร้างข้อมูลหลากหลายประเภท เช่น บทความวิชาการ, บทสนทนา, ทิปส์ท่องเที่ยว, การเงิน, เทคโนโลยี, อาหารและสุขภาพ, การเลี้ยงลูก, การแพทย์ และอื่นๆ ดูรายการชุดข้อมูลได้ที่โฟลเดอร์ <code>src/dataset/cook</code></p>
    </td>
    <td width="50%">
      <h3>🔄 Batch Generation</h3>
      <p>สร้างหลายตัวอย่างสำหรับชุดข้อมูลของคุณได้อย่างง่ายดาย รองรับการสร้างแบบ Batch และการประมวลผลแบบขนาน ช่วยให้สร้างชุดข้อมูลขนาดใหญ่ได้อย่างรวดเร็ว</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📁 Organized Output</h3>
      <p>บันทึกชุดข้อมูลที่สร้างในรูปแบบที่มีโครงสร้าง (JSON, CSV) พร้อมสำหรับการ fine-tuning models หรือ NLP pipelines อื่นๆ พร้อมข้อมูล metadata ที่สมบูรณ์</p>
    </td>
    <td width="50%">
      <h3>🧩 Extensible Architecture</h3>
      <p>ออกแบบด้วยสถาปัตยกรรมแบบโมดูลาร์ ทำให้ง่ายต่อการเพิ่มงานใหม่หรือแก้ไขงานที่มีอยู่ ด้วยระบบ config-driven ที่ยืดหยุ่น</p>
    </td>
  </tr>
</table>

## 🛠️ การติดตั้ง

ทำตามขั้นตอนเหล่านี้เพื่อติดตั้ง AiThaiBuilder บนเครื่องของคุณ:

### 1. Clone Repository

```bash
git clone https://github.com/JonusNattapong/AiThaiBuilder.git
cd AiThaiBuilder
```

### 2. ติดตั้ง Dependencies

ขอแนะนำให้ใช้ virtual environment:

```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน virtual environment
# สำหรับ Windows
venv\Scripts\activate
# สำหรับ macOS/Linux
source venv/bin/activate

# ติดตั้ง packages ที่จำเป็น
pip install -r requirements.txt
```

สำหรับการทำงานกับภาษาไทย คุณต้องติดตั้ง Thai NLP packages เพิ่มเติม:

```bash
# ติดตั้ง Thai language models สำหรับ spaCy
pip install https://github.com/explosion/spacy-models/releases/download/th_core_news_lg-3.5.0/th_core_news_lg-3.5.0.tar.gz
```

### 3. ตั้งค่า Deepseek API Key

1. รับ API key จาก [https://platform.deepseek.com/](https://platform.deepseek.com/)
2. สร้างไฟล์ `.env` ในไดเรกทอรีหลักของโปรเจค (`AiThaiBuilder/.env`) และเพิ่ม API key ของคุณ:

```env
# .env
DEEPSEEK_API_KEY="your_actual_deepseek_api_key_here"
```
## 🧑‍💻 ทีมพัฒนา

โปรเจคนี้พัฒนาและดูแลโดย:

<div align="center">
  <table>
    <tr>
      <td align="center"><a href="https://github.com/JonusNattapong"><img src="https://github.com/JonusNattapong.png" width="100px;" alt="JonusNattapong"/><br /><sub><b>JonusNattapong</b></sub></a></td>
      <td align="center"><a href="https://github.com/zombitx64"><img src="https://github.com/zombitx64.png" width="100px;" alt="zombitx64"/><br /><sub><b>zombitx64</b></sub></a></td>
    </tr>
  </table>
</div>

## 🤝 การมีส่วนร่วม

ยินดีต้อนรับการมีส่วนร่วม! หากคุณมีข้อเสนอแนะสำหรับการปรับปรุงหรือฟีเจอร์ใหม่ โปรดเปิด issue หรือส่ง pull request

### วิธีการมีส่วนร่วม

1. Fork repository
2. สร้าง feature branch (`git checkout -b feature/amazing-feature`)
3. Commit การเปลี่ยนแปลงของคุณ (`git commit -m 'Add some amazing feature'`)
4. Push ไปยัง branch (`git push origin feature/amazing-feature`)
5. เปิด Pull Request

## 📄 ลิขสิทธิ์

โปรเจคนี้อยู่ภายใต้ MIT License - ดูไฟล์ `LICENSE` สำหรับรายละเอียด

---

<p align="center">Made with ❤️ for the Thai AI Community</p>
