#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Specialized Thai Dataset Script

This script generates specialized Thai language datasets with customizable configurations.
Supports various domain-specific datasets and allows fine-grained control over content types,
complexity levels, and dataset characteristics.

Usage:
    python generate_specialized_dataset.py --domain [DOMAIN] --size [SIZE] --output [OUTPUT_FILE]
    
Example:
    python generate_specialized_dataset.py --domain medical --size 100 --output medical_dataset.json
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utils
from utils.deepseek_utils import get_deepseek_api_key, generate_with_deepseek
from utils.deepseek_translation_utils import generate_en_to_th_translation, generate_zh_to_th_translation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', f'specialized_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), 'w', 'utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

# Configuration for different domain-specific datasets
SPECIALIZED_DOMAINS = {
    "medical": {
        "description": "ชุดข้อมูลทางการแพทย์ภาษาไทย",
        "system_prompt": "คุณเป็นผู้เชี่ยวชาญด้านการแพทย์ที่มีความรู้ด้านศัพท์ทางการแพทย์ในภาษาไทย สร้างข้อมูลทางการแพทย์ที่มีความถูกต้องตามหลักวิชาการ",
        "categories": {
            "diagnosis": {
                "prompt": "สร้างบทความวินิจฉัยโรคเกี่ยวกับ {topic} ในรูปแบบที่แพทย์ใช้สื่อสารกับผู้ป่วย ใช้ศัพท์ทางการแพทย์ผสมกับภาษาทั่วไปที่เข้าใจง่าย",
                "topics": [
                    "โรคเบาหวาน", "โรคความดันโลหิตสูง", "โรคหัวใจ", "โรคไขมันในเลือดสูง", 
                    "โรคไทรอยด์", "โรคภูมิแพ้", "โรคหอบหืด", "โรคกระเพาะอาหาร", 
                    "โรคลำไส้แปรปรวน", "อาการปวดหลัง", "โรคข้อเสื่อม", "โรคกระดูกพรุน"
                ]
            },
            "treatment": {
                "prompt": "สร้างคำแนะนำการรักษาสำหรับ {topic} แบบละเอียด อธิบายวิธีการรักษาต่างๆ ทั้งการใช้ยา การผ่าตัด และการปรับเปลี่ยนวิถีชีวิต",
                "topics": [
                    "การรักษาโรคเบาหวาน", "การรักษาโรคหัวใจ", "การรักษาโรคมะเร็ง", 
                    "การรักษาโรคกระดูก", "การรักษาโรคผิวหนัง", "การรักษาโรคทางเดินหายใจ", 
                    "การรักษาโรคทางเดินอาหาร", "การรักษาโรคไต", "การรักษาโรคตับ"
                ]
            },
            "medicine": {
                "prompt": "สร้างคำอธิบายเกี่ยวกับยา {topic} โดยระบุสรรพคุณ วิธีใช้ ข้อควรระวัง และผลข้างเคียงที่อาจเกิดขึ้น",
                "topics": [
                    "ยาลดความดันโลหิต", "ยาลดไขมันในเลือด", "ยาต้านการอักเสบ", 
                    "ยาปฏิชีวนะ", "ยาแก้ปวด", "ยารักษาโรคเบาหวาน", 
                    "ยาคลายกล้ามเนื้อ", "ยาแก้แพ้", "ยานอนหลับ", "ยารักษาโรคซึมเศร้า"
                ]
            }
        }
    },
    "legal": {
        "description": "ชุดข้อมูลทางกฎหมายภาษาไทย",
        "system_prompt": "คุณเป็นนักกฎหมายที่มีความเชี่ยวชาญในกฎหมายไทย สร้างเนื้อหาทางกฎหมายที่ถูกต้องตามหลักวิชาการและกฎหมายไทยปัจจุบัน",
        "categories": {
            "contracts": {
                "prompt": "สร้างตัวอย่างสัญญา {topic} ในรูปแบบภาษากฎหมายไทยที่ถูกต้อง ครอบคลุมสาระสำคัญและเงื่อนไขทางกฎหมายที่จำเป็น",
                "topics": [
                    "สัญญาซื้อขาย", "สัญญาเช่า", "สัญญาจ้างงาน", 
                    "สัญญากู้ยืม", "สัญญาจ้างทำของ", "สัญญาตัวแทน", 
                    "สัญญาจำนอง", "สัญญาประกันภัย", "สัญญาร่วมทุน"
                ]
            },
            "laws": {
                "prompt": "อธิบาย {topic} โดยละเอียด ทั้งหลักการ เจตนารมณ์ และการบังคับใช้ในปัจจุบัน พร้อมยกตัวอย่างคดีที่เกี่ยวข้อง",
                "topics": [
                    "ประมวลกฎหมายแพ่งและพาณิชย์", "ประมวลกฎหมายอาญา", 
                    "พระราชบัญญัติคุ้มครองผู้บริโภค", "กฎหมายทรัพย์สินทางปัญญา", 
                    "กฎหมายภาษีอากร", "กฎหมายแรงงาน", "กฎหมายสิ่งแวดล้อม", 
                    "กฎหมายการแข่งขันทางการค้า", "กฎหมายล้มละลาย"
                ]
            },
            "legal_advice": {
                "prompt": "ให้คำแนะนำทางกฎหมายสำหรับกรณี {topic} โดยอธิบายสิทธิ หน้าที่ ขั้นตอนทางกฎหมาย และทางเลือกที่ผู้ประสบปัญหาควรพิจารณา",
                "topics": [
                    "การซื้อขายอสังหาริมทรัพย์", "การเรียกร้องค่าเสียหายจากอุบัติเหตุ", 
                    "การฟ้องร้องคดีผู้บริโภค", "การจดทะเบียนธุรกิจ", "การขอวีซ่าและใบอนุญาตทำงาน", 
                    "การขอสินเชื่อและการแก้ไขปัญหาหนี้", "การจัดการมรดก", 
                    "การฟ้องหย่าและการแบ่งสินสมรส", "ปัญหาการละเมิดลิขสิทธิ์"
                ]
            }
        }
    },
    "technical": {
        "description": "ชุดข้อมูลทางเทคนิคและเทคโนโลยีภาษาไทย",
        "system_prompt": "คุณเป็นผู้เชี่ยวชาญด้านเทคโนโลยีและวิศวกรรมที่มีความรู้ลึกในศัพท์เทคนิคภาษาไทย สร้างเนื้อหาทางเทคนิคที่ถูกต้องตามหลักวิชาการ",
        "categories": {
            "programming": {
                "prompt": "อธิบายเทคนิคการเขียนโปรแกรมเกี่ยวกับ {topic} โดยละเอียด พร้อมยกตัวอย่างโค้ดและคำอธิบายแนวคิดสำคัญ",
                "topics": [
                    "การเขียนโปรแกรมเชิงวัตถุ", "การพัฒนาเว็บแอปพลิเคชัน", "การพัฒนาแอปพลิเคชันมือถือ", 
                    "การจัดการฐานข้อมูล", "การทดสอบซอฟต์แวร์", "การพัฒนา API", 
                    "เทคนิคการหาและแก้ไขข้อผิดพลาด", "การพัฒนาระบบ AI", "ความปลอดภัยทางไซเบอร์"
                ]
            },
            "hardware": {
                "prompt": "อธิบายรายละเอียดทางเทคนิคของ {topic} ทั้งโครงสร้าง การทำงาน ข้อมูลจำเพาะ และการประยุกต์ใช้งาน",
                "topics": [
                    "หน่วยประมวลผลกลาง (CPU)", "การ์ดแสดงผล (GPU)", "แผงวงจรหลัก (Motherboard)", 
                    "หน่วยความจำ (RAM)", "อุปกรณ์จัดเก็บข้อมูล", "ระบบระบายความร้อน", 
                    "แหล่งจ่ายไฟ", "อุปกรณ์เครือข่าย", "อุปกรณ์อินพุต/เอาต์พุต"
                ]
            },
            "engineering": {
                "prompt": "สร้างบทความทางวิศวกรรมเกี่ยวกับ {topic} โดยอธิบายหลักการทำงาน การออกแบบ การคำนวณที่เกี่ยวข้อง และการประยุกต์ใช้ในโลกจริง",
                "topics": [
                    "วิศวกรรมโครงสร้าง", "วิศวกรรมไฟฟ้า", "วิศวกรรมเครื่องกล", 
                    "วิศวกรรมโยธา", "วิศวกรรมเคมี", "วิศวกรรมสิ่งแวดล้อม", 
                    "วิศวกรรมอุตสาหการ", "วิศวกรรมซอฟต์แวร์", "วิศวกรรมระบบ"
                ]
            }
        }
    },
    "financial": {
        "description": "ชุดข้อมูลทางการเงินและการลงทุนภาษาไทย",
        "system_prompt": "คุณเป็นนักการเงินและการลงทุนมืออาชีพที่มีความเชี่ยวชาญในภาษาทางการเงินไทย สร้างเนื้อหาทางการเงินที่ถูกต้องตามหลักวิชาการและบริบทการเงินไทย",
        "categories": {
            "investment": {
                "prompt": "อธิบายกลยุทธ์การลงทุนใน {topic} โดยครอบคลุมหลักการ ความเสี่ยง ผลตอบแทน และแนวทางการตัดสินใจลงทุนที่เหมาะสม",
                "topics": [
                    "หุ้น", "พันธบัตร", "กองทุนรวม", "อสังหาริมทรัพย์", 
                    "ทองคำ", "สกุลเงินดิจิทัล", "สินค้าโภคภัณฑ์", "การลงทุนต่างประเทศ", 
                    "กองทุน ETF", "การลงทุนเพื่อเกษียณ"
                ]
            },
            "taxes": {
                "prompt": "อธิบายระบบภาษี {topic} ในประเทศไทย โดยครอบคลุมอัตราภาษี วิธีการคำนวณ การลดหย่อน และกลยุทธ์การวางแผนภาษีที่ถูกต้องตามกฎหมาย",
                "topics": [
                    "ภาษีเงินได้บุคคลธรรมดา", "ภาษีเงินได้นิติบุคคล", "ภาษีมูลค่าเพิ่ม", 
                    "ภาษีที่ดินและสิ่งปลูกสร้าง", "ภาษีการรับมรดก", "ภาษีการโอนอสังหาริมทรัพย์", 
                    "ภาษีเงินได้จากการลงทุน", "ภาษีสำหรับธุรกิจขนาดย่อม", "ภาษีสรรพสามิต"
                ]
            },
            "banking": {
                "prompt": "สร้างบทความเกี่ยวกับผลิตภัณฑ์และบริการธนาคารเกี่ยวกับ {topic} โดยอธิบายลักษณะ ข้อดีข้อเสีย เงื่อนไข และข้อควรพิจารณาในการใช้บริการ",
                "topics": [
                    "บัญชีเงินฝากประเภทต่างๆ", "สินเชื่อบ้าน", "สินเชื่อรถยนต์", 
                    "บัตรเครดิต", "ผลิตภัณฑ์ประกันชีวิตธนาคาร", "การลงทุนผ่านธนาคาร", 
                    "บริการธนาคารดิจิทัล", "การโอนเงินระหว่างประเทศ", "สินเชื่อธุรกิจ"
                ]
            }
        }
    },
    "educational": {
        "description": "ชุดข้อมูลทางการศึกษาภาษาไทย",
        "system_prompt": "คุณเป็นนักการศึกษาที่มีความเชี่ยวชาญในการสร้างเนื้อหาการเรียนการสอนภาษาไทย สร้างเนื้อหาทางการศึกษาที่ถูกต้องตามหลักวิชาการ เข้าใจง่าย และเหมาะสมกับระดับผู้เรียน",
        "categories": {
            "sciences": {
                "prompt": "สร้างบทเรียนวิชา {topic} สำหรับนักเรียนระดับมัธยมศึกษา โดยอธิบายทฤษฎี หลักการ พร้อมตัวอย่างและแบบฝึกหัด",
                "topics": [
                    "ฟิสิกส์", "เคมี", "ชีววิทยา", "คณิตศาสตร์", 
                    "วิทยาศาสตร์คอมพิวเตอร์", "ดาราศาสตร์", "ธรณีวิทยา", 
                    "พันธุศาสตร์", "นิเวศวิทยา", "อุตุนิยมวิทยา"
                ]
            },
            "humanities": {
                "prompt": "สร้างบทเรียนวิชา {topic} สำหรับนักเรียนระดับมัธยมศึกษา โดยนำเสนอเนื้อหาที่น่าสนใจ ข้อมูลสำคัญ และประเด็นวิเคราะห์",
                "topics": [
                    "ประวัติศาสตร์ไทย", "ประวัติศาสตร์โลก", "ภูมิศาสตร์", 
                    "หน้าที่พลเมือง", "ศาสนาและจริยธรรม", "ปรัชญา", 
                    "เศรษฐศาสตร์", "ภาษาและวรรณกรรม", "ศิลปะและดนตรี"
                ]
            },
            "study_skills": {
                "prompt": "สร้างคู่มือแนะนำเทคนิคการ {topic} สำหรับนักเรียนและนักศึกษา โดยอธิบายขั้นตอน วิธีการ พร้อมเคล็ดลับที่มีประสิทธิภาพ",
                "topics": [
                    "อ่านหนังสืออย่างมีประสิทธิภาพ", "จดบันทึกในชั้นเรียน", "บริหารเวลา", 
                    "เตรียมตัวสอบ", "ทำการบ้านอย่างมีประสิทธิภาพ", "ค้นคว้าข้อมูลทางวิชาการ", 
                    "เขียนรายงานวิชาการ", "นำเสนอหน้าชั้นเรียน", "แก้ไขปัญหาการเรียน"
                ]
            }
        }
    }
}

# Default values
DEFAULT_DOMAIN = "medical"
DEFAULT_SIZE = 10
DEFAULT_OUTPUT = f"specialized_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
DEFAULT_BATCH_SIZE = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_RETRIES = 3
DEFAULT_COMPLEXITY = "medium"  # Options: simple, medium, complex

def create_prompt_for_dataset(domain, category, topic, complexity=DEFAULT_COMPLEXITY):
    """Create a prompt for dataset generation based on domain, category, and topic."""
    domain_config = SPECIALIZED_DOMAINS.get(domain, SPECIALIZED_DOMAINS[DEFAULT_DOMAIN])
    
    complexity_instructions = {
        "simple": "สร้างเนื้อหาที่เข้าใจง่าย ใช้ภาษาทั่วไป หลีกเลี่ยงศัพท์เทคนิคซับซ้อน เหมาะสำหรับบุคคลทั่วไป",
        "medium": "สร้างเนื้อหาที่มีความสมดุลระหว่างความเข้าใจง่ายและความลึกซึ้งทางวิชาการ",
        "complex": "สร้างเนื้อหาที่มีความลึกซึ้งทางวิชาการ ใช้ศัพท์เฉพาะทางอย่างถูกต้อง เหมาะสำหรับผู้เชี่ยวชาญ"
    }
    
    category_config = domain_config["categories"].get(category)
    if not category_config:
        return f"สร้างเนื้อหาเกี่ยวกับ {topic} ในด้าน {domain} ที่มีความถูกต้องและน่าเชื่อถือ {complexity_instructions.get(complexity, '')}"
    
    prompt = category_config["prompt"].format(topic=topic)
    prompt += f"\n\n{complexity_instructions.get(complexity, '')}"
    
    return prompt

def get_random_topics(domain, category, count=1):
    """Get random topics for a specific domain and category."""
    domain_config = SPECIALIZED_DOMAINS.get(domain, SPECIALIZED_DOMAINS[DEFAULT_DOMAIN])
    category_config = domain_config["categories"].get(category)
    
    if not category_config or not category_config.get("topics"):
        # Fallback topics if none are defined
        fallback_topics = [f"หัวข้อที่ {i+1} ในหมวด {category}" for i in range(10)]
        return random.sample(fallback_topics, min(count, len(fallback_topics)))
    
    topics = category_config["topics"]
    return random.sample(topics, min(count, len(topics)))

def generate_sample(domain, category, topic, api_key, complexity=DEFAULT_COMPLEXITY, max_retries=DEFAULT_MAX_RETRIES):
    """Generate a single dataset sample."""
    domain_config = SPECIALIZED_DOMAINS.get(domain, SPECIALIZED_DOMAINS[DEFAULT_DOMAIN])
    system_prompt = domain_config.get("system_prompt", "คุณเป็นผู้เชี่ยวชาญในการสร้างข้อมูลภาษาไทยที่มีคุณภาพสูง")
    user_prompt = create_prompt_for_dataset(domain, category, topic, complexity)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            content = generate_with_deepseek(messages, api_key)
            
            if content and not content.startswith("Error"):
                return {
                    "domain": domain,
                    "category": category,
                    "topic": topic,
                    "complexity": complexity,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.warning(f"Failed attempt {attempt+1}/{max_retries}: {content}")
            time.sleep(2)  # Short delay between retries
            
        except Exception as e:
            logger.error(f"Error in sample generation (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    # Return failure record after all retries
    return {
        "domain": domain,
        "category": category,
        "topic": topic,
        "complexity": complexity,
        "content": f"GENERATION_ERROR: Failed after {max_retries} attempts",
        "timestamp": datetime.now().isoformat(),
        "error": True
    }

def generate_batch(args):
    """Generate a batch of samples (for threading)."""
    domain, category, topics, api_key, complexity = args
    return [generate_sample(domain, category, topic, api_key, complexity) for topic in topics]

def generate_dataset(domain, categories, num_samples, output_file, api_key, complexity=DEFAULT_COMPLEXITY, batch_size=DEFAULT_BATCH_SIZE):
    """Generate a complete dataset."""
    if not domain or domain not in SPECIALIZED_DOMAINS:
        logger.warning(f"Domain '{domain}' not found. Using default domain: {DEFAULT_DOMAIN}")
        domain = DEFAULT_DOMAIN
        
    if not categories:
        domain_config = SPECIALIZED_DOMAINS[domain]
        categories = list(domain_config["categories"].keys())
    
    metadata = {
        "domain": domain,
        "domain_description": SPECIALIZED_DOMAINS[domain]["description"],
        "categories": categories,
        "samples_per_category": num_samples,
        "complexity": complexity,
        "date_generated": datetime.now().isoformat(),
        "generator_version": "1.0.0"
    }
    
    total_samples = len(categories) * num_samples
    all_samples = []
    
    with tqdm(total=total_samples, desc=f"Generating {domain} dataset") as pbar:
        for category in categories:
            # Distribute samples across topics
            num_topics = min(len(SPECIALIZED_DOMAINS[domain]["categories"][category].get("topics", [])), num_samples)
            if num_topics == 0:
                num_topics = 1  # Fallback to at least one topic
                
            samples_per_topic = max(1, num_samples // num_topics)
            extra_samples = num_samples % num_topics
            
            # Prepare batches for parallel processing
            batches = []
            topics_used = []
            
            for i in range(num_topics):
                topic_samples = samples_per_topic + (1 if i < extra_samples else 0)
                if topic_samples <= 0:
                    continue
                    
                topic = get_random_topics(domain, category, 1)[0]
                while topic in topics_used:  # Avoid duplicate topics
                    topic = get_random_topics(domain, category, 1)[0]
                    
                topics_used.append(topic)
                topic_list = [topic] * topic_samples
                batches.append((domain, category, topic_list, api_key, complexity))
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=min(5, len(batches))) as executor:
                batch_results = list(executor.map(generate_batch, batches))
                
                # Flatten results
                for batch in batch_results:
                    all_samples.extend(batch)
                    pbar.update(len(batch))
    
    # Check for errors
    error_samples = [s for s in all_samples if s.get("error", False)]
    if error_samples:
        logger.warning(f"{len(error_samples)} samples failed to generate properly.")
    
    # Prepare final dataset
    dataset = {
        "metadata": metadata,
        "samples": all_samples
    }
    
    # Save to file
    output_path = os.path.join(project_root, 'output', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dataset generated with {len(all_samples)} samples. Saved to {output_path}")
    return dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate specialized Thai language dataset')
    
    parser.add_argument('--domain', type=str, default=DEFAULT_DOMAIN,
                        choices=list(SPECIALIZED_DOMAINS.keys()),
                        help=f'Domain for the dataset (default: {DEFAULT_DOMAIN})')
    
    parser.add_argument('--categories', type=str, nargs='+',
                        help='Specific categories to include (default: all for the domain)')
    
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help=f'Number of samples per category (default: {DEFAULT_SIZE})')
    
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output file name (default: {DEFAULT_OUTPUT})')
    
    parser.add_argument('--complexity', type=str, default=DEFAULT_COMPLEXITY,
                        choices=['simple', 'medium', 'complex'],
                        help=f'Content complexity level (default: {DEFAULT_COMPLEXITY})')
    
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for processing (default: {DEFAULT_BATCH_SIZE})')
    
    parser.add_argument('--api-key', type=str,
                        help='Deepseek API key (will use environment variable if not provided)')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Get API key
    api_key = args.api_key or get_deepseek_api_key()
    if not api_key:
        logger.error("Deepseek API key not found. Please provide it as an argument or set it in your environment.")
        sys.exit(1)
    
    # Validate categories
    if args.categories and not all(c in SPECIALIZED_DOMAINS[args.domain]["categories"] for c in args.categories):
        invalid_categories = [c for c in args.categories if c not in SPECIALIZED_DOMAINS[args.domain]["categories"]]
        logger.warning(f"Invalid categories for domain '{args.domain}': {', '.join(invalid_categories)}")
        valid_categories = list(SPECIALIZED_DOMAINS[args.domain]["categories"].keys())
        logger.info(f"Valid categories for '{args.domain}' are: {', '.join(valid_categories)}")
        sys.exit(1)
    
    # Generate dataset
    generate_dataset(
        domain=args.domain,
        categories=args.categories,
        num_samples=args.size,
        output_file=args.output,
        api_key=api_key,
        complexity=args.complexity,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()