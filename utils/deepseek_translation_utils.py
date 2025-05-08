import os
import sys
import pandas as pd
import dask.dataframe as dd
from typing import List, Tuple, Dict, Optional
import uuid
import logging
import time
import psycopg2
from retrying import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fasttext
from tqdm import tqdm
import pickle
import gzip
import shutil
from slack_sdk.webhook import WebhookClient
import requests
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
import random

# ดาวน์โหลดข้อมูลสำหรับ METEOR
nltk.download('wordnet', quiet=True)

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# การนำเข้า deepseek_utils
try:
    from .deepseek_utils import generate_with_deepseek, get_deepseek_api_key
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

# ดาวน์โหลดโมเดล FastText ถ้าไม่มี
def download_fasttext_model(model_path: str = 'lid.176.bin'):
    if not os.path.exists(model_path):
        logger.info("Downloading FastText model lid.176.bin...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        try:
            response = requests.get(url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("FastText model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download FastText model: {str(e)}")
            raise

# โหลดโมเดล FastText
try:
    download_fasttext_model()
    fasttext_model = fasttext.load_model('lid.176.bin')
except Exception as e:
    logger.error(f"Failed to load FastText model: {str(e)}")
    raise

# ฟังก์ชันตรวจจับภาษาด้วย FastText
def detect_language_with_fasttext(text: str) -> str:
    try:
        text = text.replace('\n', ' ')
        predictions = fasttext_model.predict(text)
        lang_code = predictions[0][0].replace('__label__', '').upper()
        return lang_code
    except Exception as e:
        logger.warning(f"FastText language detection failed: {str(e)}")
        return "UNKNOWN"

# ฟังก์ชันสร้างการแปลอ้างอิง
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def generate_reference_translation(source_text: str, source_lang_code: str, target_lang_code: str, api_key: str) -> str:
    system_prompt = f"You are an expert linguist. Provide a high-quality, accurate, and natural translation of the following text."
    user_prompt = (
        f"Source Language: {source_lang_code}\n"
        f"Target Language: {target_lang_code}\n"
        f"Text: {source_text}\n"
        f"Translation:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        reference_text = generate_with_deepseek(messages, api_key)
        return reference_text
    except Exception as e:
        logger.warning(f"Reference translation failed: {str(e)}")
        return ""

# ฟังก์ชันตรวจสอบคุณภาพการแปล
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def check_translation_quality(translated_text: str, reference_text: str, api_key: str) -> Dict[str, float]:
    if not translated_text or not reference_text:
        return {'bleu': 0.0, 'rouge': 0.0, 'meteor': 0.0, 'average': 0.0}
    
    try:
        # BLEU
        reference = [reference_text.split()]
        candidate = translated_text.split()
        bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

        # ROUGE (ใช้ ROUGE-L)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_text, translated_text)
        rouge_score = rouge_scores['rougeL'].fmeasure

        # METEOR
        meteor_score_value = meteor_score([reference_text], translated_text)

        # คะแนนเฉลี่ย
        average_score = (bleu_score + rouge_score + meteor_score_value) / 3

        return {
            'bleu': bleu_score,
            'rouge': rouge_score,
            'meteor': meteor_score_value,
            'average': average_score
        }
    except Exception as e:
        logger.warning(f"Quality check failed: {str(e)}")
        return {'bleu': 0.0, 'rouge': 0.0, 'meteor': 0.0, 'average': 0.0}

# ฟังก์ชันแก้ไขการแปล
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def refine_translation(source_text: str, translated_text: str, source_lang_code: str, target_lang_code: str, api_key: str) -> str:
    system_prompt = f"You are an expert linguist. The following translation is suboptimal. Refine it to be more accurate and natural while maintaining the original meaning."
    user_prompt = (
        f"Source ({source_lang_code}): {source_text}\n"
        f"Initial Translation ({target_lang_code}): {translated_text}\n"
        f"Refined Translation:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        refined_text = generate_with_deepseek(messages, api_key)
        return refined_text
    except Exception as e:
        logger.error(f"Refinement failed: {str(e)}")
        return translated_text

# ฟังก์ชันแปล
def get_translation_prompt(source_text: str, source_lang_code: str, target_lang_code: str, additional_instructions: str = "") -> Tuple[str, str]:
    source_lang_map = {"EN": "English", "ZH": "Chinese", "HI": "Hindi"}
    target_lang_map = {"TH": "Thai"}
    source_language = source_lang_map.get(source_lang_code.upper(), source_lang_code)
    target_language = target_lang_map.get(target_lang_code.upper(), target_lang_code)

    system_prompt = f"You are an expert linguist and translator specializing in translating text from {source_language} to {target_language}. Translate accurately, naturally, and maintain the original meaning and tone."
    user_prompt = f"Please translate the following {source_language} text to {target_language}:\n\n"
    user_prompt += f"{source_language} Text:\n\"\"\"\n{source_text}\n\"\"\"\n\n"
    user_prompt += f"{target_language} Translation:"
    if additional_instructions:
        user_prompt += f"\n\nAdditional Instructions: {additional_instructions}"
    return system_prompt, user_prompt

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def generate_translation_sample(source_text: str, source_lang_code: str, target_lang_code: str, api_key: str, additional_instructions: str = "") -> str:
    if not source_text or source_text.strip() == "nan":
        return ""
    if not source_lang_code or not target_lang_code:
        logger.error("Source or target language code missing")
        return ""

    system_prompt, user_prompt = get_translation_prompt(source_text, source_lang_code, target_lang_code, additional_instructions)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        translated_text = generate_with_deepseek(messages, api_key)
        return translated_text
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise

# ฟังก์ชันจัดการแคชด้วย PostgreSQL
def init_cache_db(pg_conn_params: Dict[str, str]) -> psycopg2.extensions.connection:
    try:
        conn = psycopg2.connect(**pg_conn_params)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                translated_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        return conn
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {str(e)}")
        raise

def load_translation_cache(conn: psycopg2.extensions.connection, cache_key: str) -> Optional[str]:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT translated_text FROM translations WHERE cache_key = %s", (cache_key,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        logger.warning(f"Failed to load cache: {str(e)}")
        return None

def save_translation_cache(conn: psycopg2.extensions.connection, cache_key: str, translated_text: str, max_cache_size: int = 100000):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO translations (cache_key, translated_text) VALUES (%s, %s) ON CONFLICT (cache_key) DO UPDATE SET translated_text = %s", 
                       (cache_key, translated_text, translated_text))
        
        cursor.execute("SELECT COUNT(*) FROM translations")
        count = cursor.fetchone()[0]
        if count > max_cache_size:
            cursor.execute("DELETE FROM translations WHERE created_at = (SELECT MIN(created_at) FROM translations)")
        
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save cache: {str(e)}")

def get_cache_key(text: str, source_lang_code: str, target_lang_code: str) -> str:
    return hashlib.md5(f"{text}_{source_lang_code}_{target_lang_code}".encode('utf-8')).hexdigest()

# ฟังก์ชันส่งแจ้งเตือน Slack
def send_slack_notification(webhook_url: str, message: str):
    try:
        webhook = WebhookClient(webhook_url)
        webhook.send(text=message)
        logger.info("Slack notification sent")
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {str(e)}")

# ฟังก์ชันอ่านจาก Kafka
def read_from_kafka(kafka_config: Dict[str, str], topic: str, batch_size: int = 1000) -> pd.DataFrame:
    consumer = Consumer(kafka_config)
    consumer.subscribe([topic])
    messages = []
    try:
        while len(messages) < batch_size:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    break
            messages.append(msg.value().decode('utf-8'))
    finally:
        consumer.close()
    try:
        df = pd.DataFrame([eval(msg) for msg in messages])
        return df
    except Exception as e:
        logger.error(f"Failed to parse Kafka messages: {str(e)}")
        return pd.DataFrame()

# ฟังก์ชันประมวลผล partition
def process_partition(
    partition: pd.DataFrame,
    partition_idx: int,
    text_columns: List[str],
    source_lang_code: str,
    target_lang_code: str,
    api_key: str,
    additional_instructions: str,
    conn: psycopg2.extensions.connection,
    output_file: str,
    first_partition: bool,
    max_workers: int,
    cache_hits: List[int],
    total_texts: List[int],
    new_translations: List[int],
    quality_scores: List[Dict[str, float]],
    refinements: List[int],
    quality_threshold: float = 0.5,
    quality_check_probability: float = 0.1
):
    logger.info(f"Processing partition {partition_idx + 1} with {len(partition)} rows")
    
    translated_partition = partition.copy()
    translated_partition['thai_text'] = ""

    # รวบรวมข้อความที่ต้องแปล
    texts_to_translate = []
    indices = []
    source_texts = []
    for index, row in translated_partition.iterrows():
        for col in text_columns:
            text = str(row[col])
            if text and text.strip() != "nan":
                try:
                    detected_lang = detect_language_with_fasttext(text)
                    if detected_lang == source_lang_code.upper():
                        cache_key = get_cache_key(text, source_lang_code, target_lang_code)
                        cached_text = load_translation_cache(conn, cache_key)
                        if cached_text:
                            translated_partition.at[index, 'thai_text'] = cached_text
                            cache_hits[0] += 1
                            logger.info(f"Used cached translation for text at index {index}")
                        else:
                            texts_to_translate.append((text, source_lang_code, target_lang_code, api_key, additional_instructions, conn))
                            indices.append(index)
                            source_texts.append(text)
                        total_texts[0] += 1
                except Exception as e:
                    logger.warning(f"Language detection failed for text in row {index}, column {col}: {str(e)}")
                    translated_partition.at[index, 'thai_text'] = translated_partition.at[index, 'thai_text'] or ""
            else:
                translated_partition.at[index, 'thai_text'] = translated_partition.at[index, 'thai_text'] or ""

    # แปลแบบขนาน
    if texts_to_translate:
        logger.info(f"Translating {len(texts_to_translate)} texts in partition {partition_idx + 1}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {executor.submit(translate_text, args): args for args in texts_to_translate}
            for future in as_completed(future_to_text):
                try:
                    cache_key, translated_text = future.result()
                    index = indices[texts_to_translate.index(future_to_text[future])]
                    source_text = source_texts[indices.index(index)]
                    
                    final_translation = translated_text
                    quality_score = None
                    
                    # ตรวจสอบคุณภาพแบบสุ่มตัวอย่าง
                    if random.random() < quality_check_probability:
                        reference_text = generate_reference_translation(source_text, source_lang_code, target_lang_code, api_key)
                        if reference_text:
                            quality_score = check_translation_quality(translated_text, reference_text, api_key)
                            quality_scores.append(quality_score)
                            logger.info(f"Quality scores for text at index {index}: BLEU={quality_score['bleu']:.2f}, ROUGE={quality_score['rouge']:.2f}, METEOR={quality_score['meteor']:.2f}, Average={quality_score['average']:.2f}")
                            
                            # แก้ไขถ้าคุณภาพต่ำ
                            if quality_score['average'] < quality_threshold:
                                refined_text = refine_translation(source_text, translated_text, source_lang_code, target_lang_code, api_key)
                                refined_score = check_translation_quality(refined_text, reference_text, api_key)
                                if refined_score['average'] > quality_score['average']:
                                    final_translation = refined_text
                                    refinements[0] += 1
                                    logger.info(f"Refined translation at index {index}, new average score: {refined_score['average']:.2f}")
                    
                    translated_partition.at[index, 'thai_text'] = final_translation
                    new_translations[0] += 1
                    logger.info(f"Translated text at index {index}")
                except Exception as e:
                    logger.error(f"Translation failed in partition {partition_idx + 1}: {str(e)}")
                time.sleep(0.05)

    # บันทึก partition
    try:
        mode = 'w' if first_partition else 'a'
        header = first_partition
        translated_partition.to_csv(output_file, mode=mode, header=header, index=False, encoding='utf-8')
        logger.info(f"Saved partition {partition_idx + 1} to {output_file}")
        return None
    except Exception as e:
        logger.error(f"Failed to save partition {partition_idx + 1}: {str(e)}")
        failed_partition_file = f"failed_partition_{partition_idx}.pkl"
        with open(failed_partition_file, 'wb') as f:
            pickle.dump(translated_partition, f)
        return failed_partition_file

# ฟังก์ชันแปลแบบขนาน 
def translate_text(args: Tuple[str, str, str, str, str, psycopg2.extensions.connection]) -> Tuple[str, str]:
    text, source_lang_code, target_lang_code, api_key, additional_instructions, conn = args
    cache_key = get_cache_key(text, source_lang_code, target_lang_code)
    cached_text = load_translation_cache(conn, cache_key)
    if cached_text:
        return cache_key, cached_text
    translated_text = generate_translation_sample(text, source_lang_code, target_lang_code, api_key, additional_instructions)
    save_translation_cache(conn, cache_key, translated_text)
    return cache_key, translated_text

def auto_translate_dataset(
    input_file: Optional[str] = None,
    kafka_config: Optional[Dict[str, str]] = None,
    kafka_topic: Optional[str] = None,
    output_file: str = "translated_dataset.csv.gz",
    source_lang_code: str = "EN",
    target_lang_code: str = "TH",
    api_key: str = None,
    additional_instructions: str = "",
    columns: Optional[List[str]] = None,
    pg_conn_params: Optional[Dict[str, str]] = None,
    slack_webhook_url: Optional[str] = None,
    max_workers: int = 4,
    n_partitions: int = 4,
    max_cache_size: int = 100000,
    max_retries: int = 3,
    max_failed_partitions: int = 10,
    quality_threshold: float = 0.5,
    kafka_batch_size: int = 1000,
    quality_check_probability: float = 0.1
) -> str:
    """
    แปล dataset อัตโนมัติโดยใช้ Dask, FastText, PostgreSQL, Kafka, และเมตริก BLEU, ROUGE, METEOR

    Args:
        input_file (Optional[str]): พาธไปยังไฟล์ dataset (CSV)
        kafka_config (Optional[Dict[str, str]]): การกำหนดค่า Kafka
        kafka_topic (Optional[str]): Kafka topic สำหรับสตรีม
        output_file (str): พาธสำหรับบันทึก dataset ที่แปลแล้ว (.csv.gz)
        source_lang_code (str): รหัสภาษาต้นฉบับที่ต้องการแปล
        target_lang_code (str): รหัสภาษาเป้าหมาย
        api_key (str): Deepseek API key
        additional_instructions (str): คำสั่งเพิ่มเติมสำหรับการแปล
        columns (Optional[List[str]]): รายการคอลัมน์ที่ต้องการแปล
        pg_conn_params (Optional[Dict[str, str]]): พารามิเตอร์การเชื่อมต่อ PostgreSQL
        slack_webhook_url (Optional[str]): Slack Webhook URL
        max_workers (int): จำนวน thread สูงสุด
        n_partitions (int): จำนวน partition สำหรับ Dask
        max_cache_size (int): จำนวนรายการสูงสุดในแคช
        max_retries (int): จำนวนครั้งที่ลองใหม่
        max_failed_partitions (int): จำนวน partition ที่ล้มเหลวสูงสุด
        quality_threshold (float): เกณฑ์คะแนนเฉลี่ยสำหรับการแก้ไข
        kafka_batch_size (int): ขนาด batch สำหรับ Kafka
        quality_check_probability (float): ความน่าจะเป็นในการตรวจสอบคุณภาพ
    """
    start_time = time.time()
    if not api_key:
        api_key = get_deepseek_api_key()

    if not (input_file or (kafka_config and kafka_topic)):
        logger.error("Must provide either input_file or kafka_config with kafka_topic")
        return "Error: ต้องระบุ input_file หรือ kafka_config กับ kafka_topic"

    # ลบไฟล์ชั่วคราวเก่า
    temp_output_file = output_file.replace('.csv.gz', '.temp.csv') if output_file.endswith('.csv.gz') else output_file + '.temp.csv'
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
    for f in os.listdir():
        if f.startswith("failed_partition_") and f.endswith(".pkl"):
            os.remove(f)

    # เริ่มต้นแคช PostgreSQL
    if pg_conn_params is None:
        pg_conn_params = {
            'dbname': 'translation_db',
            'user': 'user',
            'password': 'password',
            'host': 'localhost',
            'port': '5432'
        }
    try:
        conn = init_cache_db(pg_conn_params)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        return f"Error: ไม่สามารถเชื่อมต่อ PostgreSQL: {str(e)}"

    # ตัวนับสำหรับสถิติ
    cache_hits = [0]
    total_texts = [0]
    new_translations = [0]
    quality_scores = []
    refinements = [0]

    # อ่าน dataset
    partitions = []
    if input_file:
        try:
            ddf = dd.read_csv(input_file, blocksize=25e6)
            ddf = ddf.repartition(n_partitions=n_partitions)
            logger.info(f"Loaded dataset with {n_partitions} partitions from {input_file}")
            partitions = [(i, ddf.get_partition(i).compute()) for i in range(n_partitions)]
        except Exception as e:
            logger.error(f"Failed to read file {input_file}: {str(e)}")
            conn.close()
            return f"Error: ไม่สามารถอ่านไฟล์ {input_file}. รายละเอียด: {str(e)}"
    elif kafka_config and kafka_topic:
        try:
            df = read_from_kafka(kafka_config, kafka_topic, kafka_batch_size)
            if df.empty:
                logger.error("No data received from Kafka")
                conn.close()
                return "Error: ไม่มีข้อมูลจาก Kafka"
            ddf = dd.from_pandas(df, npartitions=n_partitions)
            logger.info(f"Loaded dataset with {n_partitions} partitions from Kafka topic {kafka_topic}")
            partitions = [(i, ddf.get_partition(i).compute()) for i in range(n_partitions)]
        except Exception as e:
            logger.error(f"Failed to read from Kafka: {str(e)}")
            conn.close()
            return f"Error: ไม่สามารถอ่านจาก Kafka: {str(e)}"

    # เก็บ partition ที่ล้มเหลว
    failed_partitions = []
    first_partition = True

    # ประมวลผล partition
    for partition_idx, partition in tqdm(partitions, desc="Processing partitions"):
        try:
            text_columns = columns if columns else partition.select_dtypes(include=['object']).columns.tolist()
            if not text_columns:
                logger.warning(f"No text columns found in partition {partition_idx + 1}")
                continue

            result = process_partition(
                partition, partition_idx, text_columns, source_lang_code, target_lang_code,
                api_key, additional_instructions, conn, temp_output_file, first_partition,
                max_workers, cache_hits, total_texts, new_translations, quality_scores,
                refinements, quality_threshold, quality_check_probability
            )
            if result:
                failed_partitions.append(result)
                if len(failed_partitions) > max_failed_partitions:
                    logger.error(f"Too many failed partitions: {failed_partitions}")
                    conn.close()
                    return f"Error: มี partition ล้มเหลวเกิน {max_failed_partitions}: {failed_partitions}"
            first_partition = False
        except Exception as e:
            logger.error(f"Failed to process partition {partition_idx + 1}: {str(e)}")
            failed_partition_file = f"failed_partition_{partition_idx}.pkl"
            with open(failed_partition_file, 'wb') as f:
                pickle.dump(partition, f)
            failed_partitions.append(failed_partition_file)

    # ลองประมวลผล partition ที่ล้มเหลวใหม่
    retries = 0
    while failed_partitions and retries < max_retries:
        retries += 1
        logger.info(f"Retrying failed partitions (attempt {retries}/{max_retries})")
        new_failed_partitions = []
        for failed_partition_file in failed_partitions:
            try:
                with open(failed_partition_file, 'rb') as f:
                    partition = pickle.load(f)
                partition_idx = int(failed_partition_file.split('_')[-1].split('.')[0])
                text_columns = columns if columns else partition.select_dtypes(include=['object']).columns.tolist()
                
                result = process_partition(
                    partition, partition_idx, text_columns, source_lang_code, target_lang_code,
                    api_key, additional_instructions, conn, temp_output_file, False,
                    max_workers, cache_hits, total_texts, new_translations, quality_scores,
                    refinements, quality_threshold, quality_check_probability
                )
                if result:
                    new_failed_partitions.append(result)
                else:
                    os.remove(failed_partition_file)
            except Exception as e:
                logger.error(f"Retry failed for partition {failed_partition_file}: {str(e)}")
                new_failed_partitions.append(failed_partition_file)
        failed_partitions = new_failed_partitions

    # บีบอัดไฟล์เป็น gzip
    if os.path.exists(temp_output_file):
        try:
            with open(temp_output_file, 'rb') as f_in:
                with gzip.open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(temp_output_file)
            logger.info(f"Compressed output to {output_file}")
        except Exception as e:
            logger.error(f"Failed to compress output: {str(e)}")
            failed_partitions.append("compression_failed")

    # ปิดการเชื่อมต่อ PostgreSQL
    conn.close()

    # สร้างข้อความแจ้งเตือน
    elapsed_time = time.time() - start_time
    avg_bleu = sum(s['bleu'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_rouge = sum(s['rouge'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_meteor = sum(s['meteor'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_score = sum(s['average'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    summary = (
        f"Translation Summary\n"
        f"Status: {'Success' if not failed_partitions else 'Failed'}\n"
        f"Total texts: {total_texts[0]}\n"
        f"Cache hits: {cache_hits[0]}\n"
        f"New translations: {new_translations[0]}\n"
        f"Refined translations: {refinements[0]}\n"
        f"Average BLEU score: {avg_bleu:.2f}\n"
        f"Average ROUGE-L score: {avg_rouge:.2f}\n"
        f"Average METEOR score: {avg_meteor:.2f}\n"
        f"Average combined score: {avg_score:.2f} (from {len(quality_scores)} samples)\n"
        f"Time: {elapsed_time:.2f}s\n"
        f"Output: {output_file}\n"
    )
    if failed_partitions:
        summary += f"Failed partitions: {failed_partitions}\n"

    # ส่งแจ้งเตือน Slack
    if slack_webhook_url:
        send_slack_notification(slack_webhook_url, summary)

    if failed_partitions:
        logger.error(f"Failed partitions after {max_retries} retries: {failed_partitions}")
        return f"Error: บาง partition ล้มเหลว: {failed_partitions}"

    logger.info(summary)
    return f"แปลสำเร็จ! ผลลัพธ์บันทึกที่ {output_file}"

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    input_file = "input_dataset.csv"
    kafka_config = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'translation_group',
        'auto.offset.reset': 'earliest'
    }
    kafka_topic = "translation_topic"
    output_file = "translated_dataset.csv.gz"
    api_key = get_deepseek_api_key()
    source_lang_code = "EN"
    columns = ["english_text"]
    pg_conn_params = {
        'dbname': 'translation_db',
        'user': 'user',
        'password': 'password',
        'host': 'localhost',
        'port': '5432'
    }
    slack_webhook_url = "https://hooks.slack.com/services/xxx/yyy/zzz"  # แทนที่ด้วย Webhook URL จริง

    result = auto_translate_dataset(
        input_file=input_file,
        kafka_config=kafka_config,
        kafka_topic=kafka_topic,
        output_file=output_file,
        source_lang_code=source_lang_code,
        target_lang_code="TH",
        api_key=api_key,
        columns=columns,
        pg_conn_params=pg_conn_params,
        slack_webhook_url=slack_webhook_url,
        n_partitions=4,
        max_cache_size=100000,
        max_retries=3,
        max_failed_partitions=10,
        quality_threshold=0.5,
        kafka_batch_size=1000,
        quality_check_probability=0.1
    )
    print(result)