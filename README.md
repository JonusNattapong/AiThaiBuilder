<p align="center">
  <img src="https://github.com/JonusNattapong/RunThaiGenDataset/assets/banner.png" alt="RunThaiGenDataset Banner" width="600"/>
</p>

<h1 align="center">RunThaiGenDataset</h1>
<p align="center">🤖 เครื่องมือสร้างชุดข้อมูลภาษาไทยคุณภาพสูง 🤖</p>

<p align="center">
  <a href="https://github.com/JonusNattapong/RunThaiGenDataset/blob/main/LICENSE"><img src="https://img.shields.io/github/license/JonusNattapong/RunThaiGenDataset?color=blue" alt="License"></a>
  <a href="https://github.com/JonusNattapong/RunThaiGenDataset/stargazers"><img src="https://img.shields.io/github/stars/JonusNattapong/RunThaiGenDataset?color=yellow" alt="Stars"></a>
  <a href="https://github.com/JonusNattapong/RunThaiGenDataset/network/members"><img src="https://img.shields.io/github/forks/JonusNattapong/RunThaiGenDataset?color=green" alt="Forks"></a>
  <a href="https://github.com/JonusNattapong/RunThaiGenDataset/issues"><img src="https://img.shields.io/github/issues/JonusNattapong/RunThaiGenDataset?color=red" alt="Issues"></a>
</p>

<p align="center">
  <b>RunThaiGenDataset</b> คือเครื่องมือออกแบบมาสำหรับสร้างชุดข้อมูลภาษาไทยสำหรับ NLP ที่มีความหลากหลาย 
  ด้วยการใช้ Deepseek API เพื่อการสร้างข้อความที่มีคุณภาพและให้อินเตอร์เฟซที่ใช้งานง่ายด้วย Gradio
</p>

---

## 📚 สารบัญ
- [✨ คุณลักษณะเด่น](#-คุณลักษณะเด่น)
- [🛠️ การติดตั้ง](#️-การติดตั้ง)
- [🚀 วิธีการใช้งาน](#-วิธีการใช้งาน)
- [📖 API Documentation](#-api-documentation-conceptual)
- [🧑‍💻 ทีมพัฒนา](#-ทีมพัฒนา)
- [🤝 การมีส่วนร่วม](#-การมีส่วนร่วม)
- [📄 ลิขสิทธิ์](#-ลิขสิทธิ์)

---

## ✨ คุณลักษณะเด่น

<table>
  <tr>
    <td width="50%">
      <h3>🔄 Dataset Generation แบบหลากหลาย</h3>
      <ul>
        <li>Text Generation</li>
        <li>Summarization</li>
        <li>Question Answering</li>
        <li>Translation (EN-TH, ZH-TH)</li>
        <li>Instruction Following</li>
        <li>และอื่นๆ อีกมากมาย (ดูเพิ่มเติมได้ที่ <code>config/config.json</code>)</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🤖 Powered by Deepseek API</h3>
      <p>ใช้ language models ที่ทันสมัยที่สุดเพื่อสร้างข้อความภาษาไทยที่มีความเกี่ยวข้องกับบริบทและมีความสอดคล้องกัน</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>💬 Customizable Prompts</h3>
      <p>ผู้ใช้สามารถกำหนด system prompts และคำแนะนำเพิ่มเติมเพื่อปรับแต่งกระบวนการสร้างได้ตามต้องการ</p>
    </td>
    <td width="50%">
      <h3>📊 Dataset Quality Analysis</h3>
      <p>มีเครื่องมือสำหรับวิเคราะห์และตรวจสอบคุณภาพของชุดข้อมูลพร้อมรายงานและการแสดงผลที่ครอบคลุม</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📚 หลากหลายประเภทข้อมูล</h3>
      <p>รองรับการสร้างข้อมูลหลากหลายประเภท เช่น บทความวิชาการ, บทสนทนา, ทิปส์ท่องเที่ยว, การเงิน, เทคโนโลยี, อาหารและสุขภาพ, การเลี้ยงลูก, และอื่นๆ อีกมากมาย</p>
    </td>
    <td width="50%">
      <h3>🔄 Batch Generation</h3>
      <p>สร้างหลายตัวอย่างสำหรับชุดข้อมูลของคุณได้อย่างง่ายดาย</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📁 Organized Output</h3>
      <p>บันทึกชุดข้อมูลที่สร้างในรูปแบบที่มีโครงสร้าง พร้อมสำหรับการ fine-tuning models หรือ NLP pipelines อื่นๆ</p>
    </td>
    <td width="50%">
      <h3>🧩 Extensible</h3>
      <p>ออกแบบด้วยการกำหนดค่าแบบโมดูลาร์ ทำให้ง่ายต่อการเพิ่มงานใหม่หรือแก้ไขงานที่มีอยู่</p>
    </td>
  </tr>
</table>

## 🛠️ การติดตั้ง

ทำตามขั้นตอนเหล่านี้เพื่อติดตั้ง RunThaiGenDataset บนเครื่องของคุณ:

### 1. Clone Repository:

```bash
git clone https://github.com/your-username/RunThaiGenDataset.git
cd RunThaiGenDataset
```

### 2. ติดตั้ง Dependencies:

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

### 3. ตั้งค่า Deepseek API Key:

1. รับ API key จาก [https://platform.deepseek.com/](https://platform.deepseek.com/)
2. สร้างไฟล์ `.env` ในไดเรกทอรีหลักของโปรเจค (`RunThaiGenDataset/.env`) และเพิ่ม API key ของคุณ:

```env
# .env
DEEPSEEK_API_KEY="your_actual_deepseek_api_key_here"
```

หรือคุณสามารถป้อน API key โดยตรงในอินเตอร์เฟซ Gradio เมื่อรันแอปพลิเคชัน

## 🚀 วิธีการใช้งาน

เมื่อการติดตั้งเสร็จสมบูรณ์ คุณสามารถเริ่มแอปพลิเคชัน Gradio:

```bash
python src/app.py
```

ไปที่ URL ที่แสดงในเทอร์มินัลของคุณ (โดยปกติคือ `http://127.0.0.1:7860`) โดยใช้เว็บเบราว์เซอร์ของคุณเพื่อเข้าถึง Thai Dataset Generator

## 📖 API Documentation (Conceptual)

โปรเจคนี้มี OpenAPI specification (`openapi.yaml`) ที่อธิบาย API เชิงแนวคิดสำหรับฟังก์ชันการสร้างชุดข้อมูล ซึ่งมีประโยชน์สำหรับความเข้าใจโครงสร้างข้อมูลและพารามิเตอร์ที่เกี่ยวข้อง

คุณสามารถดูเอกสารนี้โดยใช้ **ReDoc**

### การดูด้วย ReDoc:

1. **การใช้ Docker container (แนะนำ):**  
   หากคุณมี Docker ติดตั้งแล้ว คุณสามารถให้บริการอินเตอร์เฟซ ReDoc ได้อย่างรวดเร็ว:

   ```bash
   # สำหรับ PowerShell:
   docker run -p 8080:80 -v ${PWD}\openapi.yaml:/usr/share/nginx/html/openapi.yaml redocly/redoc

   # สำหรับ CMD (ตรวจสอบว่าอยู่ในไดเรกทอรีโปรเจค):
   docker run -p 8080:80 -v "%CD%\openapi.yaml":/usr/share/nginx/html/openapi.yaml redocly/redoc

   # สำหรับ Git Bash หรือ Linux terminals:
   docker run -p 8080:80 -v $(pwd)/openapi.yaml:/usr/share/nginx/html/openapi.yaml redocly/redoc
   ```
   
   จากนั้นเปิด `http://localhost:8080` ในเบราว์เซอร์ของคุณ มันควรจะโหลด `openapi.yaml` โดยอัตโนมัติ หากไม่ลองใช้ `http://localhost:8080/?url=openapi.yaml`

2. **การใช้ online viewer:**  
   คุณสามารถใช้ online ReDoc viewer โดยอัปโหลดหรือวางเนื้อหาของ `openapi.yaml` ลงในเครื่องมือ
   ตัวอย่าง: [https://redocly.github.io/redoc/](https://redocly.github.io/redoc/)  
   *(หมายเหตุ: ระวังการวางข้อมูลที่ละเอียดอ่อนหรือ schemas ที่เป็นกรรมสิทธิ์ลงในเครื่องมือออนไลน์สาธารณะ)*

3. **การใช้ `redoc-cli` (ต้องมี Node.js):**  
   หากคุณมี Node.js และ npm ติดตั้งแล้ว:

   ```bash
   # ติดตั้ง redoc-cli แบบ global (ทำเพียงครั้งเดียว)
   npm install -g redoc-cli

   # นำทางไปยังไดเรกทอรีโปรเจคของคุณ
   # cd path/to/RunThaiGenDataset

   # ให้บริการเอกสาร
   redoc-cli serve openapi.yaml
   ```

   โดยปกติแล้วจะเปิดเอกสารในเบราว์เซอร์ของคุณที่ `http://127.0.0.1:8080/`

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

## 📄 ลิขสิทธิ์

โปรเจคนี้อยู่ภายใต้ MIT License - ดูไฟล์ `LICENSE` สำหรับรายละเอียด

---

<p align="center">Made with ❤️ for the Thai NLP community</p>