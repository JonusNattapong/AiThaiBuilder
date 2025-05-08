<p align="center">
  <img src="assets/banner.png" alt="RunThaiGenDataset Banner" width="600"/>
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
- [📊 การวิเคราะห์ข้อมูล](#-การวิเคราะห์ข้อมูล)
- [📚 โครงสร้างชุดข้อมูล](#-โครงสร้างชุดข้อมูล)
- [🔮 แนวทางการพัฒนาในอนาคต](#-แนวทางการพัฒนาในอนาคต)
- [📖 API Documentation](#-api-documentation)
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

ทำตามขั้นตอนเหล่านี้เพื่อติดตั้ง RunThaiGenDataset บนเครื่องของคุณ:

### 1. Clone Repository

```bash
git clone https://github.com/JonusNattapong/RunThaiGenDataset.git
cd RunThaiGenDataset
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
2. สร้างไฟล์ `.env` ในไดเรกทอรีหลักของโปรเจค (`RunThaiGenDataset/.env`) และเพิ่ม API key ของคุณ:

```env
# .env
DEEPSEEK_API_KEY="your_actual_deepseek_api_key_here"
```

## 🚀 วิธีการใช้งาน

RunThaiGenDataset มีเครื่องมือหลายตัวสำหรับการสร้างและจัดการข้อมูลภาษาไทย ต่อไปนี้เป็นตัวอย่างพื้นฐานในการใช้งาน:

### 1. การสร้างชุดข้อมูลพื้นฐาน

```bash
# สร้างชุดข้อมูล Sentiment Analysis
python src/script/genarate/generate_text_classification_dataset.py --task sentiment --samples-per-label 10 --output-file output/sentiment_dataset.json

# สร้างชุดข้อมูลภาษาถิ่น
python src/script/genarate/generate_thai_nlp_dataset.py --dataset-type dialect --samples 30 --categories northern_thai southern_thai --output-file output/dialect_dataset.json
```

### 2. การสร้างชุดข้อมูลเฉพาะทาง

```bash
# สร้างชุดข้อมูลทางการแพทย์
python src/script/genarate/specialized/generate_specialized_dataset.py --domain medical --size 50 --output output/medical_dataset.json

# สร้างชุดข้อมูลทางกฎหมาย
python src/script/genarate/specialized/generate_specialized_dataset.py --domain legal --size 50 --categories contracts laws --output output/legal_dataset.json
```

### 3. การสร้างชุดข้อมูลสำหรับงาน NLP ขั้นสูง

```bash
# สร้างชุดข้อมูล Named Entity Recognition (NER)
python src/script/genarate/generate_ner_dataset.py --samples 50 --domains ข่าว การเมือง การแพทย์ --output-file output/ner_dataset.json

# สร้างชุดข้อมูล Coreference Resolution
python src/script/genarate/generate_coreference_resolution_dataset.py --samples 30 --domains ข่าว บันเทิง การเมือง --output-file output/coref_dataset.json

# สร้างชุดข้อมูล Semantic Role Labeling
python src/script/genarate/generate_semantic_role_labeling_dataset.py --samples 30 --domains เศรษฐกิจ กีฬา วิทยาศาสตร์ --output-file output/srl_dataset.json
```

### 4. การทำความสะอาดและเตรียมข้อมูล

```bash
# ทำความสะอาดข้อมูล
python src/script/clean/clean_dataset.py --input output/sentiment_dataset.json --output processed/sentiment_dataset_cleaned.csv

# รวมชุดข้อมูลหลายชุด
python src/script/combine/combine_datasets.py --input output/sentiment_basic_*.json output/sentiment_complex_*.json --output processed/sentiment_combined.csv
```

### 5. การ Fine-tuning โมเดล

```bash
# Fine-tune โมเดลสำหรับ Sentiment Analysis
python src/script/train/run_fine_tuning.py \
  --model wangchanberta \
  --model_size base \
  --task classification \
  --input_file processed/sentiment_dataset_cleaned.csv \
  --text_column text \
  --label_column sentiment \
  --output_dir model/save/sentiment_model \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epochs 5
```

### 6. การแปลชุดข้อมูล

```bash
# แปลชุดข้อมูลจากภาษาอังกฤษเป็นภาษาไทย
python src/script/genarate/specialized/translate_dataset_inplace.py --input dataset_en.json --source-lang en --target-lang th --fields text title
```

## 📊 การวิเคราะห์ข้อมูล

RunThaiGenDataset มีเครื่องมือวิเคราะห์ข้อมูลเพื่อตรวจสอบคุณภาพและลักษณะของชุดข้อมูลของคุณ:

### 1. วิเคราะห์ข้อมูลเบื้องต้น

```bash
# วิเคราะห์ข้อมูลทั่วไป
python src/script/analyze/run_data_analysis.py --input_file processed/sentiment_dataset.csv --text_column text --label_column sentiment
```

### 2. ตรวจสอบคุณภาพข้อมูลขั้นสูง

```bash
# ตรวจสอบคุณภาพข้อมูลแบบละเอียด
python src/script/check/check_dataset_quality.py --input_file processed/sentiment_dataset.csv --advanced

# ตรวจสอบความซ้ำซ้อนของข้อมูล
python src/script/check/check_dataset_duplicates.py --input_file processed/combined_dataset.csv --text_column text
```

### 3. วิเคราะห์ข้อมูลเฉพาะทาง

```bash
# วิเคราะห์ชุดข้อมูลภาษาถิ่น
python src/script/check/check_dataset_quality_dialect.py --input_file output/dialect_dataset.csv --dialect_column dialect

# วิเคราะห์ความสอดคล้องของการติดป้าย NER
python src/script/check/check_ner_consistency.py --input_file processed/ner_dataset.csv
```

ผลการวิเคราะห์จะถูกบันทึกในโฟลเดอร์ `analysis_output/` พร้อมรายงาน PDF และกราฟต่างๆ เช่น:

- คุณภาพข้อความและการกระจายตัว
- สถิติความยาวและความซับซ้อนของประโยค
- การกระจายตัวของคลาสและป้ายกำกับ
- ข้อมูลทาง Part-of-Speech และอื่นๆ

## 📚 โครงสร้างชุดข้อมูล

โปรเจคนี้มีโครงสร้างชุดข้อมูลที่หลากหลายสำหรับงาน NLP ภาษาไทย โดยเก็บไว้ในโฟลเดอร์ `src/dataset/cook` แบ่งเป็นหมวดหมู่ต่างๆ ดังนี้:

### 1. ชุดข้อมูลตามประเภทภาษาถิ่น

- **central_thai_dataset.py**: ภาษาไทยกลาง
- **northern_thai_dataset.py**: ภาษาไทยเหนือ
- **northeastern_thai_dataset.py**: ภาษาไทยอีสาน
- **southern_thai_dataset.py**: ภาษาไทยใต้

### 2. ชุดข้อมูลตามประเภทอารมณ์

- **happy_dataset.py**: ข้อความแสดงความสุข
- **sad_dataset.py**: ข้อความแสดงความเศร้า
- **angry_dataset.py**: ข้อความแสดงความโกรธ
- **fear_dataset.py**: ข้อความแสดงความกลัว

### 3. ชุดข้อมูลเฉพาะทาง

- **medical_dataset.py**: ข้อมูลทางการแพทย์
- **legal_dataset.py**: ข้อมูลทางกฎหมาย
- **technical_dataset.py**: ข้อมูลทางเทคนิค
- **educational_dataset.py**: ข้อมูลทางการศึกษา
- **financial_dataset.py**: ข้อมูลทางการเงิน

### 4. ชุดข้อมูลตามประเภทงาน NLP

- **qa_dataset.py**: ชุดคำถาม-คำตอบ
- **conversation_dataset.py**: บทสนทนา
- **summarization_dataset.py**: การสรุปความ
- **translation_dataset.py**: การแปลภาษา
- **academic_dataset.py**: บทความวิชาการ
- **code_explanation_dataset.py**: คำอธิบายโค้ด

## 🔮 แนวทางการพัฒนาในอนาคต

RunThaiGenDataset มีแผนการพัฒนาอย่างต่อเนื่องเพื่อเพิ่มความสามารถและประสิทธิภาพ ดังนี้:

### 1. การ Fine-tune โมเดลขั้นสูง

* **รองรับ Parameter-Efficient Fine-Tuning (PEFT):** เพิ่มการรองรับเทคนิคเช่น LoRA, QLoRA เพื่อให้สามารถ fine-tune โมเดลขนาดใหญ่ได้อย่างมีประสิทธิภาพมากขึ้นโดยใช้ทรัพยากรน้อยลง
- **Hyperparameter Optimization:** ผนวกเครื่องมือสำหรับการค้นหา Hyperparameters ที่ดีที่สุดโดยอัตโนมัติ (เช่น Optuna, Ray Tune)
- **Experiment Tracking:** เชื่อมต่อกับแพลตฟอร์มเช่น Weights & Biases (W&B) หรือ MLflow เพื่อการติดตามและเปรียบเทียบผลการทดลองที่สะดวกยิ่งขึ้น
- **รองรับโมเดลใหม่:** เพิ่มการรองรับโมเดลใหม่ล่าสุดเช่น sAnti-2 และโมเดลภาษาไทยอื่นๆ

### 2. การสร้างและปรับปรุงชุดข้อมูล

* **เพิ่มความหลากหลายของภาษาถิ่น:** ขยายชุดข้อมูลภาษาถิ่นให้ครอบคลุมภูมิภาคและกลุ่มชาติพันธุ์ต่างๆ ในประเทศไทยมากขึ้น
- **เทคนิค Data Augmentation ใหม่ๆ:** ค้นคว้าและเพิ่มเทคนิคการเพิ่มข้อมูล (Data Augmentation) ที่ทันสมัยและเหมาะสมกับบริบทภาษาไทย
- **Controllable Generation:** พัฒนาความสามารถในการควบคุมคุณลักษณะของข้อมูลที่สร้างขึ้น เช่น รูปแบบภาษา, อารมณ์, หรือเนื้อหาเฉพาะทาง
- **ปรับปรุงคุณภาพข้อมูลที่สร้าง:** พัฒนากลไกในการประเมินและปรับปรุงคุณภาพของข้อมูลที่สร้างจาก AI ให้มีความเป็นธรรมชาติและถูกต้องตามหลักภาษา

### 3. การตรวจสอบคุณภาพชุดข้อมูล

* **ตัวชี้วัดคุณภาพขั้นสูง:** เพิ่มเมตริกการประเมินคุณภาพชุดข้อมูลที่ซับซ้อนยิ่งขึ้น เช่น การตรวจจับความเอนเอียง (Bias Detection) ในระดับลึก, Semantic Drift, และความสอดคล้องของข้อมูล (Data Consistency)
- **รายงานผลแบบโต้ตอบ:** พัฒนารูปแบบการแสดงผลการวิเคราะห์คุณภาพชุดข้อมูลให้เป็นแบบ Interactive Dashboard เพื่อให้ผู้ใช้สามารถสำรวจข้อมูลได้ง่ายขึ้น
- **Fairness Analysis:** เพิ่มระบบตรวจสอบความเป็นธรรมในชุดข้อมูลเพื่อลดอคติทางเพศ, อายุ, หรือสถานะทางสังคม

### 4. โครงสร้างพื้นฐานและส่วนสนับสนุน

* **เอกสารและคู่มือการใช้งาน:** ปรับปรุงเอกสารให้ละเอียด, เข้าใจง่าย, และครอบคลุมทุกฟังก์ชัน พร้อมตัวอย่างการใช้งาน (Tutorials)
- **ระบบทดสอบอัตโนมัติ (Automated Testing):** เพิ่ม Unit Test และ Integration Test ให้ครอบคลุมทุกส่วนของโปรเจ็ค
- **CI/CD Pipeline:** สร้างระบบ Continuous Integration/Continuous Deployment (CI/CD) เพื่อการพัฒนาที่รวดเร็วและมีคุณภาพ
- **การเผยแพร่เป็น Package:** พิจารณาการสร้าง Package เพื่อให้ง่ายต่อการติดตั้งและนำไปใช้งานในโปรเจ็คอื่นๆ
- **API Service:** พัฒนา REST API สำหรับการสร้างชุดข้อมูลแบบออนไลน์

## 📖 API Documentation

โปรเจคนี้มี OpenAPI specification (`openapi.yaml`) ที่อธิบาย API เชิงแนวคิดสำหรับฟังก์ชันการสร้างชุดข้อมูล ซึ่งมีประโยชน์สำหรับความเข้าใจโครงสร้างข้อมูลและพารามิเตอร์ที่เกี่ยวข้อง

คุณสามารถดูเอกสารนี้โดยใช้ **ReDoc**

### การดูด้วย ReDoc

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

### วิธีการมีส่วนร่วม

1. Fork repository
2. สร้าง feature branch (`git checkout -b feature/amazing-feature`)
3. Commit การเปลี่ยนแปลงของคุณ (`git commit -m 'Add some amazing feature'`)
4. Push ไปยัง branch (`git push origin feature/amazing-feature`)
5. เปิด Pull Request

## 📄 ลิขสิทธิ์

โปรเจคนี้อยู่ภายใต้ MIT License - ดูไฟล์ `LICENSE` สำหรับรายละเอียด

---

<p align="center">Made with ❤️ for the Thai NLP community</p>
