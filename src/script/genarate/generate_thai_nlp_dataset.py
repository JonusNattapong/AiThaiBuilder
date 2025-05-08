import os
import json
import random
import time
import argparse
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Deepseek utilities
from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key
from utils.deepseek_translation_utils import generate_en_to_th_translation

# Load environment variables from .env file
load_dotenv()

# Constants for API configuration
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 20
BATCH_SIZE = 10
OUTPUT_DIR = 'output'

# Base text generation parameters
TEXT_GENERATION_PARAMS = {
    'temperature': 0.7,
    'max_tokens': 150,
    'top_p': 0.9,
    'frequency_penalty': 0.5,
    'presence_penalty': 0.5
}

# Dataset types and their configurations
DATASET_TYPES = {
    'sentiment': {
        'description': 'ข้อความแสดงความรู้สึกหรืออารมณ์ต่างๆ',
        'categories': {
            'basic_emotions': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่แสดงอารมณ์พื้นฐาน
ให้เน้น:
1. สร้างประโยคภาษาไทยที่เป็นธรรมชาติ สมจริง
2. ใช้คำที่สื่ออารมณ์อย่างชัดเจนตรงไปตรงมา
3. ความยาว 1-2 ประโยค กระชับได้ใจความ
4. เขียนในรูปแบบบทสนทนาหรือความคิดที่พบได้ในชีวิตประจำวัน""",
                'labels': {
                    'happy': ['มีความสุข', 'ดีใจ', 'สนุกสนาน', 'ยินดี', 'ตื่นเต้น'],
                    'sad': ['เศร้า', 'เสียใจ', 'ผิดหวัง', 'ท้อแท้', 'หดหู่'],
                    'angry': ['โกรธ', 'หงุดหงิด', 'ฉุนเฉียว', 'โมโห', 'เดือดดาล']
                }
            },
            'complex_emotions': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่แสดงอารมณ์ซับซ้อน
ให้เน้น:
1. สร้างประโยคที่สื่อความรู้สึกละเอียดอ่อน ลึกซึ้ง
2. ผสมผสานอารมณ์หลายแง่มุมอย่างเป็นธรรมชาติ
3. ใช้ภาษาไทยที่สละสลวย มีชั้นเชิง
4. สร้างเรื่องราวหรือสถานการณ์ที่สมจริง น่าเชื่อถือ""",
                'labels': {
                    'love': ['รัก', 'หลงใหล', 'เสน่หา', 'ผูกพัน', 'อบอุ่น'],
                    'fear': ['กลัว', 'วิตก', 'หวาดกลัว', 'ตื่นตระหนก', 'หวั่นใจ'],
                    'surprise': ['ประหลาดใจ', 'ตกใจ', 'อัศจรรย์ใจ', 'พิศวง', 'งงงวย']
                }
            },
            'social_emotions': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยที่แสดงอารมณ์ทางสังคม
ให้เน้น:
1. ใช้สำนวนภาษาที่สอดคล้องกับวัฒนธรรมไทย
2. สะท้อนค่านิยมและบรรทัดฐานทางสังคมไทย
3. เลือกใช้ระดับภาษาให้เหมาะสมกับสถานการณ์
4. นำเสนอการปฏิสัมพันธ์ระหว่างบุคคลอย่างสมจริง""",
                'labels': {
                    'gratitude': ['ขอบคุณ', 'ซาบซึ้ง', 'สำนึกบุญคุณ', 'กตัญญู', 'ประทับใจ'],
                    'pride': ['ภูมิใจ', 'มั่นใจ', 'เชิดหน้าชูตา', 'สง่างาม', 'ยโส'],
                    'shame': ['อับอาย', 'ละอายใจ', 'เขินอาย', 'หน้าแตก', 'ขายหน้า']
                }
            }
        },
        'topics': [
            'ครอบครัว', 'ความรัก', 'มิตรภาพ', 'การทำงาน', 'การเรียน',
            'กีฬา', 'ดนตรี', 'ภาพยนตร์', 'สุขภาพ', 'การเงิน',
            'สังคม', 'การเมือง', 'สิ่งแวดล้อม', 'การเดินทาง', 'ความฝัน',
            'อาหาร', 'การท่องเที่ยว', 'เทคโนโลยี', 'ศิลปะ', 'วัฒนธรรม'
        ],
        'examples': {
            'happy': ['เรียนจบแล้ว! ดีใจจนน้ำตาไหล ความพยายามของเราไม่สูญเปล่า', 'ได้เจอเพื่อนเก่าโดยบังเอิญ คิดถึงจนกอดกันร้องไห้เลย'],
            'sad': ['ดูแลต้นไม้มาตั้งนาน วันนี้มันเหี่ยวตายซะแล้ว ใจหายจัง', 'เก็บของเก่าแล้วเจอรูปถ่ายครั้งสุดท้ายกับคุณยาย น้ำตาไหลโดยไม่รู้ตัว'],
            'angry': ['บอกแล้วว่าอย่าเอาขยะมาทิ้งหน้าบ้าน ยังทำอีก โมโหจริงๆ!', 'รถติดเพราะคนจอดซ้อนคัน แถมยังเถียงว่าตัวเองถูก หงุดหงิดที่สุด'],
            'love': ['ทุกครั้งที่เธอยิ้ม ฉันรู้สึกเหมือนโลกทั้งใบสว่างไสว', 'แค่ได้นั่งข้างกันโดยไม่ต้องพูดอะไร ก็รู้สึกอบอุ่นหัวใจแล้ว'],
            'fear': ['ได้ยินเสียงแปลกๆ จากห้องน้ำตอนดึก หัวใจเต้นรัวไปหมด', 'วันนี้ต้องสอบพรีเซนต์ที่สำคัญมาก มือไม้สั่นไปหมดแล้ว'],
            'surprise': ['เปิดประตูเข้าบ้านแล้วเจอเพื่อนๆ จัดปาร์ตี้วันเกิดให้ ตกใจจนพูดไม่ออก', 'อ่านข่าวเช้านี้แล้วอึ้ง ไม่คิดว่าจะเกิดเหตุการณ์แบบนี้ขึ้นจริงๆ'],
            'gratitude': ['ต้องขอบคุณที่คอยอยู่เคียงข้างกันในวันที่ยากลำบาก จะไม่มีวันลืมบุญคุณนี้', 'ไม่รู้จะตอบแทนยังไงที่ช่วยดูแลลูกให้ตอนที่ป่วย ซาบซึ้งใจจริงๆ'],
            'pride': ['ลูกสอบติดมหาวิทยาลัยที่ใฝ่ฝัน ความพยายามของเรามีค่าที่สุด', 'ผลงานชิ้นนี้ได้รับการยกย่องจากผู้เชี่ยวชาญ ภูมิใจที่สุดในชีวิต'],
            'shame': ['พูดผิดต่อหน้าลูกค้าสำคัญ อยากให้แผ่นดินสูบ', 'ลืมปิดไมค์ตอนประชุมออนไลน์ แล้วพูดไม่ดีออกไป อายจนไม่กล้าเจอใคร']
        }
    },
    'dialect': {
        'description': 'ข้อความในภาษาถิ่นต่างๆ ของไทย',
        'categories': {
            'central_thai': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยกลาง 
ให้สร้างข้อความที่:
1. เป็นภาษาไทยกลางที่ใช้ในชีวิตประจำวัน
2. สะท้อนวัฒนธรรมและคำศัพท์เฉพาะของภาคกลาง
3. มีความเป็นธรรมชาติ สมจริง""",
                'examples': [
                    'เย็นนี้ไปเดินตลาดนัดจตุจักรกันมั้ย',
                    'ข้าวผัดกะเพราหมูกรอบอร่อยสุดๆ',
                    'วันหยุดนี้จะไปไหว้พระที่วัดพระแก้ว',
                    'ซื้อของฝากจากเยาวราชมาให้ด้วยนะ',
                    'ไปดูหนังที่เมเจอร์รัชโยธินกันไหม'
                ]
            },
            'northern_thai': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยถิ่นเหนือ (คำเมือง)
ให้สร้างข้อความที่:
1. ใช้คำศัพท์ สำนวน และโครงสร้างประโยคของภาษาถิ่นเหนือ
2. สะท้อนวัฒนธรรมและวิถีชีวิตของชาวเหนือ
3. มีความเป็นธรรมชาติ เสมือนคนท้องถิ่นพูดจริง""",
                'examples': [
                    'ข้าวเหนียวมะม่วงเจ้านี้หวานหอมหลาย',
                    'ไปดูการฟ้อนเล็บที่งานสงกรานต์เน้อ',
                    'ชาเขียวจากดอยแม่สลองนี่ล่ะหอมชื่นใจ',
                    'ไปเดินป่าที่ดอยอินทนนท์กันบ่',
                    'ขนมจีนน้ำเงี้ยวเจ้านี้อร่อยคักเลย'
                ]
            },
            'northeastern_thai': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยถิ่นอีสาน
ให้สร้างข้อความที่:
1. ใช้คำศัพท์ สำนวน และโครงสร้างประโยคของภาษาถิ่นอีสาน
2. สะท้อนวัฒนธรรมและวิถีชีวิตของชาวอีสาน
3. มีความเป็นธรรมชาติ เสมือนคนท้องถิ่นพูดจริง""",
                'examples': [
                    'ไปดูบั้งไฟที่ริมโขงคืนนี้',
                    'วันนี้สิไปใส่บาตรที่วัดพระธาตุ',
                    'กินข้าวหลามย่างไฟนี่ล่ะหอมอร่อย',
                    'ไปซื้อของที่ตลาดท่าแร่กัน',
                    'พิธีผูกแขนนี่ล่ะคือประเพณีดีงาม'
                ]
            },
            'southern_thai': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างข้อความภาษาไทยถิ่นใต้
ให้สร้างข้อความที่:
1. ใช้คำศัพท์ สำนวน และโครงสร้างประโยคของภาษาถิ่นใต้
2. สะท้อนวัฒนธรรมและวิถีชีวิตของชาวใต้
3. มีความเป็นธรรมชาติ เสมือนคนท้องถิ่นพูดจริง""",
                'examples': [
                    'โรตีแกงเขียวหวานไก่นี่อร่อยคัก',
                    'ไปดำน้ำดูปะการังที่เกาะหลีเป๊ะบ่',
                    'หอยนางรมสดๆ จากสุราษฎร์นี่ล่ะสุดยอด',
                    'ไปงานชักพระที่พัทลุงกันบ่',
                    'ข้าวเหนียวทุเรียนนี่ล่ะหวานมัน'
                ]
            }
        },
        'topics': [
            'อาหาร', 'การท่องเที่ยว', 'ประเพณี', 'วิถีชีวิต', 'ครอบครัว',
            'การเกษตร', 'ศาสนา', 'เทศกาล', 'ธรรมชาติ', 'การละเล่น',
            'ตลาด', 'เพลงพื้นบ้าน', 'การแต่งกาย', 'การเดินทาง', 'สิ่งของเครื่องใช้'
        ]
    },
    'academic': {
        'description': 'บทความทางวิชาการในหัวข้อต่างๆ',
        'categories': {
            'education': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างบทความวิชาการด้านการศึกษาภาษาไทย
ให้เขียนบทความที่:
1. มีเนื้อหาเชิงวิชาการ ใช้ศัพท์เฉพาะทาง
2. มีการอ้างอิงแนวคิดและทฤษฎีที่เกี่ยวข้อง
3. นำเสนอประเด็นด้านการศึกษาอย่างเป็นระบบ
4. ใช้ภาษาเป็นทางการ สละสลวย""",
                'topics': [
                    'การเรียนรู้แบบ Active Learning ช่วยกระตุ้นความคิด',
                    'วิทยาศาสตร์ข้อมูลเป็นทักษะที่จำเป็นในยุคใหม่',
                    'การเขียนบทความวิชาการต้องอ้างอิงแหล่งข้อมูลที่น่าเชื่อถือ',
                    'การศึกษาระยะยาวแสดงผลเชิงพฤติกรรมที่ชัดเจน',
                    'ผลการทดลองควรทำซ้ำได้ในสภาพแวดล้อมเดียวกัน'
                ]
            },
            'science': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างบทความวิชาการด้านวิทยาศาสตร์ภาษาไทย
ให้เขียนบทความที่:
1. มีเนื้อหาเชิงวิทยาศาสตร์ ใช้ศัพท์เฉพาะทาง
2. มีการอธิบายปรากฏการณ์ทางวิทยาศาสตร์อย่างเป็นระบบ
3. ให้ข้อมูลที่ถูกต้องตามหลักวิชาการ
4. ใช้ภาษาเป็นทางการ ชัดเจน เข้าใจง่าย""",
                'topics': [
                    'การเปลี่ยนแปลงสภาพภูมิอากาศส่งผลต่อระบบนิเวศ',
                    'แบตเตอรี่ของอนาคตจะเปลี่ยนโลกพลังงานอย่างไร',
                    'ระบบประสาทและสมองทำงานประสานกันอย่างซับซ้อน',
                    'ปัญญาประดิษฐ์กับการพัฒนาชีวิตมนุษย์',
                    'ทฤษฎีสัมพัทธภาพและการประยุกต์ใช้ในปัจจุบัน'
                ]
            },
            'environment': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างบทความวิชาการด้านสิ่งแวดล้อมภาษาไทย
ให้เขียนบทความที่:
1. มีเนื้อหาเกี่ยวกับประเด็นสิ่งแวดล้อม ใช้ศัพท์เฉพาะทาง
2. นำเสนอสถานการณ์ปัญหาและแนวทางแก้ไขอย่างเป็นระบบ
3. อ้างอิงข้อมูลเชิงวิทยาศาสตร์และสถิติที่เกี่ยวข้อง
4. ใช้ภาษาเป็นทางการ ชัดเจน เข้าใจง่าย""",
                'topics': [
                    'ภาวะโลกร้อนและการเปลี่ยนแปลงสภาพภูมิอากาศ',
                    'การอนุรักษ์ความหลากหลายทางชีวภาพ',
                    'การจัดการขยะพลาสติกในมหาสมุทร',
                    'พลังงานหมุนเวียนกับการพัฒนาที่ยั่งยืน',
                    'ปัญหามลพิษทางอากาศในเมืองใหญ่'
                ]
            }
        }
    },
    'conversation': {
        'description': 'บทสนทนาโต้ตอบในหัวข้อและสถานการณ์ต่างๆ',
        'categories': {
            'daily_conversation': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างบทสนทนาภาษาไทยในชีวิตประจำวัน
ให้สร้างบทสนทนาที่:
1. มีความเป็นธรรมชาติ สมจริง
2. มีการโต้ตอบระหว่างบุคคลตั้งแต่ 2 คนขึ้นไป
3. แสดงบริบทและอารมณ์ของผู้สนทนาอย่างชัดเจน
4. ใช้ภาษาที่เหมาะสมกับสถานการณ์""",
                'scenarios': [
                    'การสั่งอาหารในร้านอาหาร',
                    'การถามทาง',
                    'การสนทนาในงานสังสรรค์',
                    'การพูดคุยกับเพื่อนร่วมงาน',
                    'การซื้อของในตลาด'
                ]
            },
            'business_conversation': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างบทสนทนาภาษาไทยในบริบทธุรกิจ
ให้สร้างบทสนทนาที่:
1. มีความเป็นมืออาชีพ เหมาะสมกับบริบทธุรกิจ
2. มีการใช้ศัพท์เฉพาะทางธุรกิจอย่างถูกต้อง
3. แสดงมารยาทและจรรยาบรรณทางธุรกิจ
4. ใช้ภาษาที่สุภาพ เป็นทางการในระดับที่เหมาะสม""",
                'scenarios': [
                    'การประชุมทางธุรกิจ',
                    'การเจรจาต่อรองราคา',
                    'การสัมภาษณ์งาน',
                    'การนำเสนอโครงการต่อลูกค้า',
                    'การให้คำปรึกษาด้านการลงทุน'
                ]
            },
            'customer_service': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างบทสนทนาภาษาไทยในบริบทงานบริการลูกค้า
ให้สร้างบทสนทนาที่:
1. แสดงทักษะการบริการลูกค้าที่ดี
2. มีการรับมือกับสถานการณ์ต่างๆ อย่างมืออาชีพ
3. แสดงการแก้ไขปัญหาและการตอบสนองความต้องการของลูกค้า
4. ใช้ภาษาที่สุภาพ เป็นมิตร ให้ความช่วยเหลือ""",
                'scenarios': [
                    'การรับเรื่องร้องเรียนจากลูกค้า',
                    'การให้ความช่วยเหลือทางโทรศัพท์',
                    'การแนะนำสินค้าให้ลูกค้า',
                    'การแก้ไขปัญหาสินค้าชำรุด',
                    'การตอบคำถามเกี่ยวกับบริการ'
                ]
            }
        }
    },
    'qa': {
        'description': 'คำถาม-คำตอบในหัวข้อต่างๆ',
        'categories': {
            'general_knowledge': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างชุดคำถาม-คำตอบความรู้ทั่วไปภาษาไทย
ให้สร้างชุดคำถาม-คำตอบที่:
1. ครอบคลุมความรู้ทั่วไปที่น่าสนใจ
2. มีคำตอบที่ถูกต้อง ชัดเจน
3. ให้ข้อมูลที่เป็นประโยชน์
4. มีความหลากหลายของระดับความยาก""",
                'topics': [
                    'ประวัติศาสตร์', 'วิทยาศาสตร์', 'ภูมิศาสตร์', 'ศิลปะ', 'วัฒนธรรม',
                    'เทคโนโลยี', 'กีฬา', 'บันเทิง', 'การเมือง', 'เศรษฐกิจ'
                ]
            },
            'technical_qa': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างชุดคำถาม-คำตอบด้านเทคนิคภาษาไทย
ให้สร้างชุดคำถาม-คำตอบที่:
1. เกี่ยวข้องกับความรู้เชิงเทคนิคเฉพาะทาง
2. มีคำอธิบายที่ชัดเจน ถูกต้องตามหลักวิชาการ
3. ใช้ศัพท์เฉพาะทางอย่างเหมาะสม
4. ให้ข้อมูลที่เป็นประโยชน์ต่อผู้เรียนรู้""",
                'topics': [
                    'การเขียนโปรแกรม', 'วิศวกรรมซอฟต์แวร์', 'ปัญญาประดิษฐ์', 'วิทยาการข้อมูล',
                    'เครือข่ายคอมพิวเตอร์', 'ความปลอดภัยไซเบอร์', 'อิเล็กทรอนิกส์', 'วิศวกรรมโยธา'
                ]
            },
            'interview_qa': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างชุดคำถาม-คำตอบสำหรับการสัมภาษณ์งานภาษาไทย
ให้สร้างชุดคำถาม-คำตอบที่:
1. เกี่ยวข้องกับการสัมภาษณ์งานในสาขาต่างๆ
2. มีคำตอบตัวอย่างที่ดี มีประสิทธิภาพ
3. แสดงทักษะและคุณสมบัติที่ผู้สัมภาษณ์ต้องการ
4. ให้เทคนิคและข้อแนะนำในการตอบคำถาม""",
                'topics': [
                    'ตำแหน่งผู้จัดการ', 'งานด้านการตลาด', 'งานด้านเทคโนโลยี', 'งานด้านบริการลูกค้า',
                    'งานด้านการเงิน', 'งานด้านการศึกษา', 'งานด้านสุขภาพ', 'งานด้านการโรงแรม'
                ]
            }
        }
    },
    'code_explanation': {
        'description': 'คำอธิบายโค้ดในภาษาโปรแกรมต่างๆ',
        'categories': {
            'python': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างคำอธิบายโค้ดภาษา Python เป็นภาษาไทย
ให้สร้างโค้ดและคำอธิบายที่:
1. เป็นโค้ดที่ทำงานได้จริง ถูกต้องตามหลักการเขียนโปรแกรม
2. มีคำอธิบายที่ชัดเจน เข้าใจง่าย
3. อธิบายแนวคิดและหลักการที่สำคัญ
4. เหมาะสำหรับผู้เรียนที่มีระดับความรู้ตั้งแต่เริ่มต้นจนถึงขั้นสูง""",
                'topics': [
                    'การจัดการข้อมูลด้วย Pandas', 'การสร้างกราฟด้วย Matplotlib',
                    'การประมวลผลภาษาธรรมชาติด้วย NLTK', 'การสร้าง API ด้วย FastAPI',
                    'Machine Learning ด้วย Scikit-learn'
                ]
            },
            'javascript': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างคำอธิบายโค้ดภาษา JavaScript เป็นภาษาไทย
ให้สร้างโค้ดและคำอธิบายที่:
1. เป็นโค้ดที่ทำงานได้จริง ถูกต้องตามหลักการเขียนโปรแกรม
2. มีคำอธิบายที่ชัดเจน เข้าใจง่าย
3. อธิบายแนวคิดและหลักการที่สำคัญ
4. เหมาะสำหรับผู้เรียนที่มีระดับความรู้ตั้งแต่เริ่มต้นจนถึงขั้นสูง""",
                'topics': [
                    'การสร้าง Single Page Application ด้วย React', 'การใช้ Async/Await',
                    'การจัดการ DOM', 'การสร้าง API ด้วย Node.js',
                    'การใช้ ES6 Features'
                ]
            },
            'sql': {
                'prompt': """คุณคือ AI ที่เชี่ยวชาญในการสร้างคำอธิบายคำสั่ง SQL เป็นภาษาไทย
ให้สร้างคำสั่งและคำอธิบายที่:
1. เป็นคำสั่งที่ทำงานได้จริง ถูกต้องตามหลักการ
2. มีคำอธิบายที่ชัดเจน เข้าใจง่าย
3. อธิบายแนวคิดและหลักการที่สำคัญ
4. เหมาะสำหรับผู้เรียนที่มีระดับความรู้ตั้งแต่เริ่มต้นจนถึงขั้นสูง""",
                'topics': [
                    'การสร้างและจัดการฐานข้อมูล', 'การใช้ JOIN', 'การใช้ Subqueries',
                    'การสร้าง Stored Procedures', 'การปรับแต่งประสิทธิภาพ (Query Optimization)'
                ]
            }
        }
    }
}

def create_prompt_for_dataset(dataset_type, category, label=None, topic=None, scenario=None):
    """
    สร้าง prompt สำหรับ dataset ประเภทต่างๆ
    """
    dataset_config = DATASET_TYPES[dataset_type]
    category_config = dataset_config['categories'][category]
    
    if dataset_type == 'sentiment':
        # สร้าง prompt สำหรับ sentiment dataset
        examples = dataset_config['examples'].get(label, ["", ""])
        return f"""สร้างประโยคภาษาไทยที่แสดงอารมณ์ "{label}" เกี่ยวกับ "{topic}"

ตัวอย่างประโยคที่แสดงอารมณ์นี้:
1. {examples[0]}
2. {examples[1]}

กรุณาสร้างประโยคใหม่ที่:
1. ใช้ภาษาไทยที่เป็นธรรมชาติ
2. สื่ออารมณ์ได้ชัดเจน
3. เกี่ยวข้องกับหัวข้อที่กำหนด
4. ไม่ซ้ำกับตัวอย่าง

ความยาว: 1-2 ประโยค
ตอบ:"""
    
    elif dataset_type == 'dialect':
        # สร้าง prompt สำหรับ dialect dataset
        examples = '\n'.join([f"{i+1}. {ex}" for i, ex in enumerate(category_config['examples'][:3])])
        
        return f"""สร้างประโยคในภาษาถิ่น {category.replace('_', ' ')} เกี่ยวกับหัวข้อ "{topic}"

ตัวอย่างประโยคในภาษาถิ่นนี้:
{examples}

กรุณาสร้างประโยคใหม่ที่:
1. ใช้คำศัพท์และสำนวนของภาษาถิ่นนี้อย่างถูกต้อง
2. เกี่ยวข้องกับหัวข้อที่กำหนด
3. สะท้อนวัฒนธรรมและวิถีชีวิตของท้องถิ่น
4. เป็นธรรมชาติ เหมือนคนท้องถิ่นพูดจริงๆ

ความยาว: 1-2 ประโยค
ตอบเฉพาะประโยคภาษาถิ่น:"""
    
    elif dataset_type == 'academic':
        # สร้าง prompt สำหรับ academic dataset
        return f"""เขียนบทความวิชาการภาษาไทยสั้นๆ เกี่ยวกับหัวข้อ: "{topic}"

กรุณาเขียนบทความที่:
1. มีความถูกต้องทางวิชาการ
2. มีการอ้างอิงแนวคิดหรือทฤษฎีที่เกี่ยวข้อง
3. มีการวิเคราะห์ประเด็นสำคัญ
4. ใช้ภาษาทางวิชาการที่เหมาะสม

ความยาว: ประมาณ 3-5 ย่อหน้า
บทความ:"""
    
    elif dataset_type == 'conversation':
        # สร้าง prompt สำหรับ conversation dataset
        return f"""สร้างบทสนทนาภาษาไทยในสถานการณ์: "{scenario}"

กรุณาสร้างบทสนทนาที่:
1. มีผู้สนทนาอย่างน้อย 2 คน (ระบุชื่อหรือบทบาท)
2. มีความเป็นธรรมชาติ สมจริง
3. แสดงการโต้ตอบที่เหมาะสมในสถานการณ์
4. มีบริบทและอารมณ์ที่ชัดเจน

ความยาว: ประมาณ 8-12 บรรทัด
รูปแบบ: ระบุชื่อผู้พูด ตามด้วยเครื่องหมาย ":" และข้อความ

บทสนทนา:"""
    
    elif dataset_type == 'qa':
        # สร้าง prompt สำหรับ qa dataset
        return f"""สร้างชุดคำถาม-คำตอบภาษาไทยเกี่ยวกับ "{topic}"

กรุณาสร้างชุดคำถาม-คำตอบที่:
1. มีความถูกต้อง น่าเชื่อถือ
2. มีคำอธิบายที่ชัดเจน เข้าใจง่าย
3. ให้ความรู้ที่เป็นประโยชน์
4. มีความลึกซึ้งในระดับที่เหมาะสม

โปรดสร้าง 1 คำถามพร้อมคำตอบในรูปแบบ:
คำถาม: [คำถามเกี่ยวกับหัวข้อที่กำหนด]
คำตอบ: [คำตอบที่ถูกต้อง ครบถ้วน]"""
    
    elif dataset_type == 'code_explanation':
        # สร้าง prompt สำหรับ code explanation dataset
        return f"""สร้างโค้ดตัวอย่างและคำอธิบายในภาษา {category} เกี่ยวกับ "{topic}"

กรุณาสร้างเนื้อหาที่:
1. มีโค้ดที่ทำงานได้จริง ถูกต้องตามหลักการเขียนโปรแกรม
2. อธิบายแนวคิดและการทำงานของโค้ดเป็นภาษาไทย
3. ให้ความรู้ที่เป็นประโยชน์และประยุกต์ใช้ได้
4. เหมาะสำหรับผู้เรียนระดับกลาง

รูปแบบ:
# หัวข้อ: [หัวข้อย่อย]
# คำอธิบาย: [คำอธิบายสั้นๆ]

```{category}
[โค้ดตัวอย่าง]
```

## คำอธิบายโดยละเอียด
[คำอธิบายแนวคิด หลักการ และการทำงานของโค้ดโดยละเอียด]"""
    
    return None

def generate_samples_with_deepseek(prompts, api_key, system_prompts=None, max_retries=3, delay=1):
    """
    Generate samples using Deepseek API in batch with retries
    
    Args:
        prompts (list): List of prompts to generate text for
        api_key (str): Deepseek API key
        system_prompts (list, optional): List of system prompts corresponding to each prompt
        max_retries (int, optional): Maximum number of retries for failed requests
        delay (int, optional): Delay between retries in seconds
        
    Returns:
        list: Generated texts
    """
    if system_prompts is None:
        system_prompts = ["คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญในการสร้างข้อมูลภาษาไทย"] * len(prompts)
    
    results = [None] * len(prompts)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        for i, (system_prompt, user_prompt) in enumerate(zip(system_prompts, prompts)):
            # Define the task as a function that will be executed in a separate thread
            def process_prompt(idx, retry_count=0):
                if retry_count >= max_retries:
                    print(f"Failed to process prompt {idx} after {max_retries} retries")
                    return
                
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = generate_with_deepseek(
                        messages, 
                        api_key,
                        max_tokens=TEXT_GENERATION_PARAMS['max_tokens'],
                        temperature=TEXT_GENERATION_PARAMS['temperature']
                    )
                    
                    if response and not response.startswith("Error:"):
                        results[idx] = response
                    else:
                        if retry_count < max_retries - 1:
                            time.sleep(delay)
                            process_prompt(idx, retry_count + 1)
                        else:
                            print(f"Failed to generate content for prompt {idx}: {response}")
                            results[idx] = f"GENERATION_ERROR: {response}"
                except Exception as e:
                    if retry_count < max_retries - 1:
                        time.sleep(delay)
                        process_prompt(idx, retry_count + 1)
                    else:
                        print(f"Exception processing prompt {idx}: {str(e)}")
                        results[idx] = f"GENERATION_ERROR: {str(e)}"
            
            # Submit the task to the executor
            futures.append(executor.submit(process_prompt, i))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()
    
    return results

def generate_dataset(dataset_type, output_file, num_samples, api_key, categories=None):
    """
    Generate dataset for a specific type
    
    Args:
        dataset_type (str): Dataset type (sentiment, dialect, academic, etc.)
        output_file (str): Path to save the output file
        num_samples (int): Number of samples to generate per category/label
        api_key (str): Deepseek API key
        categories (list, optional): Specific categories to generate (default: all)
    """
    if dataset_type not in DATASET_TYPES:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        return
    
    dataset_config = DATASET_TYPES[dataset_type]
    all_categories = list(dataset_config['categories'].keys())
    
    if categories is None:
        categories = all_categories
    else:
        # Validate categories
        invalid_categories = [cat for cat in categories if cat not in all_categories]
        if invalid_categories:
            print(f"Warning: Unknown categories for {dataset_type}: {', '.join(invalid_categories)}")
            categories = [cat for cat in categories if cat in all_categories]
    
    if not categories:
        print(f"Error: No valid categories specified for {dataset_type}")
        return
    
    generated_samples = []
    system_prompts = []
    user_prompts = []
    metadata = []
    
    # Prepare generation prompts
    if dataset_type == 'sentiment':
        for category in categories:
            category_config = dataset_config['categories'][category]
            for label in category_config['labels']:
                for _ in range(num_samples):
                    topic = random.choice(dataset_config['topics'])
                    system_prompts.append(category_config['prompt'])
                    user_prompts.append(create_prompt_for_dataset(dataset_type, category, label=label, topic=topic))
                    metadata.append({'category': category, 'label': label, 'topic': topic})
    
    elif dataset_type == 'dialect':
        for category in categories:
            category_config = dataset_config['categories'][category]
            for _ in range(num_samples):
                topic = random.choice(dataset_config['topics'])
                system_prompts.append(category_config['prompt'])
                user_prompts.append(create_prompt_for_dataset(dataset_type, category, topic=topic))
                metadata.append({'category': category, 'dialect': category, 'topic': topic})
    
    elif dataset_type == 'academic':
        for category in categories:
            category_config = dataset_config['categories'][category]
            for topic in category_config['topics']:
                for _ in range(num_samples):
                    system_prompts.append(category_config['prompt'])
                    user_prompts.append(create_prompt_for_dataset(dataset_type, category, topic=topic))
                    metadata.append({'category': category, 'topic': topic})
    
    elif dataset_type == 'conversation':
        for category in categories:
            category_config = dataset_config['categories'][category]
            for scenario in category_config['scenarios']:
                for _ in range(num_samples):
                    system_prompts.append(category_config['prompt'])
                    user_prompts.append(create_prompt_for_dataset(dataset_type, category, scenario=scenario))
                    metadata.append({'category': category, 'scenario': scenario})
    
    elif dataset_type == 'qa':
        for category in categories:
            category_config = dataset_config['categories'][category]
            for topic in category_config['topics']:
                for _ in range(num_samples):
                    system_prompts.append(category_config['prompt'])
                    user_prompts.append(create_prompt_for_dataset(dataset_type, category, topic=topic))
                    metadata.append({'category': category, 'topic': topic})
    
    elif dataset_type == 'code_explanation':
        for category in categories:
            category_config = dataset_config['categories'][category]
            for topic in category_config['topics']:
                for _ in range(num_samples):
                    system_prompts.append(category_config['prompt'])
                    user_prompts.append(create_prompt_for_dataset(dataset_type, category, topic=topic))
                    metadata.append({'category': category, 'programming_language': category, 'topic': topic})
    
    # Generate samples
    total_samples = len(user_prompts)
    if total_samples == 0:
        print(f"Error: No samples to generate for {dataset_type}")
        return
    
    print(f"Generating {total_samples} samples for {dataset_type} dataset")
    
    with tqdm(total=total_samples, desc=f"Generating {dataset_type} samples") as pbar:
        # Process in batches
        batch_size = min(BATCH_SIZE, total_samples)
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_prompts = user_prompts[i:end_idx]
            batch_system_prompts = system_prompts[i:end_idx]
            batch_metadata = metadata[i:end_idx]
            
            batch_results = generate_samples_with_deepseek(batch_prompts, api_key, batch_system_prompts)
            
            for result, meta in zip(batch_results, batch_metadata):
                if result and not result.startswith("GENERATION_ERROR"):
                    sample = {
                        'content': result,
                        'dataset_type': dataset_type,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        **meta
                    }
                    generated_samples.append(sample)
                else:
                    print(f"Error generating sample: {result}")
            
            pbar.update(len(batch_prompts))
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully generated {len(generated_samples)} samples for {dataset_type} dataset")
    print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Thai NLP datasets using Deepseek API')
    parser.add_argument('--dataset-type', type=str, required=True, choices=list(DATASET_TYPES.keys()),
                      help='Type of dataset to generate')
    parser.add_argument('--output-file', type=str, default=None,
                      help='Output file path (default: output/[dataset_type]_[timestamp].json)')
    parser.add_argument('--samples', type=int, default=10,
                      help='Number of samples per category/label')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                      help='Specific categories to generate (default: all)')
    parser.add_argument('--api-key', type=str, default=None,
                      help='Deepseek API key (will use environment variable if not provided)')
    
    args = parser.parse_args()
    
    # Load API key
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: Deepseek API key not found. Please provide it with --api-key or set DEEPSEEK_API_KEY in your .env file.")
        return
    
    # Set default output file if not provided
    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(OUTPUT_DIR, f"thai_{args.dataset_type}_dataset_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate dataset
    generate_dataset(args.dataset_type, args.output_file, args.samples, api_key, args.categories)

if __name__ == "__main__":
    main()