import os
import sys
import pandas as pd
import dask.dataframe as dd
from typing import List, Tuple, Dict, Optional
import logging
import time
from retrying import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import gzip
import shutil
import requests
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
import random
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Download data for METEOR
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import deepseek_utils
try:
    from .deepseek_utils import generate_with_deepseek, get_deepseek_api_key
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

# Translation functions
def get_translation_prompt(source_text: str, source_lang_code: str, target_lang_code: str, additional_instructions: str = "") -> Tuple[str, str]:
    source_lang_map = {
        "EN": "English",
        "ZH": "Chinese",
        "HI": "Hindi"  # เพิ่ม Hindi
    }
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

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def detect_language_with_deepseek(text: str, api_key: str) -> str:
    if not text or text.strip() == "nan":
        return "UNKNOWN"
    
    system_prompt = "You are a language detection expert. Identify the language of the given text and return the ISO 639-1 language code (e.g., 'EN' for English, 'ZH' for Chinese, 'HI' for Hindi, 'TH' for Thai)."
    user_prompt = f"Text: {text}\nLanguage Code:"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        lang_code = generate_with_deepseek(messages, api_key).strip().upper()
        return lang_code
    except Exception as e:
        logger.warning(f"Deepseek language detection failed: {str(e)}")
        return "UNKNOWN"

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

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def check_translation_quality(
    translated_text: str, 
    reference_text: str, 
    api_key: str, 
    metric_weights: Dict[str, float] = {'bleu': 0.3, 'rouge': 0.3, 'meteor': 0.4}
) -> Dict[str, float]:
    if not translated_text or not reference_text:
        return {'bleu': 0.0, 'rouge': 0.0, 'meteor': 0.0, 'average': 0.0}
    
    try:
        # BLEU
        reference = [reference_text.split()]
        candidate = translated_text.split()
        bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

        # ROUGE (use ROUGE-L)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_text, translated_text)
        rouge_score = rouge_scores['rougeL'].fmeasure

        # METEOR
        meteor_score_value = meteor_score([reference_text], translated_text)

        # Weighted average score
        average_score = (
            metric_weights['bleu'] * bleu_score +
            metric_weights['rouge'] * rouge_score +
            metric_weights['meteor'] * meteor_score_value
        ) / sum(metric_weights.values())

        return {
            'bleu': bleu_score,
            'rouge': rouge_score,
            'meteor': meteor_score_value,
            'average': average_score
        }
    except Exception as e:
        logger.warning(f"Quality check failed: {str(e)}")
        return {'bleu': 0.0, 'rouge': 0.0, 'meteor': 0.0, 'average': 0.0}

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def refine_translation(
    source_text: str, 
    translated_text: str, 
    source_lang_code: str, 
    target_lang_code: str, 
    api_key: str, 
    quality_scores: Dict[str, float]
) -> str:
    system_prompt = f"You are an expert linguist. The following translation is suboptimal. Refine it to be more accurate, natural, and maintain the original meaning."
    
    refinement_instructions = []
    if quality_scores['bleu'] < 0.5:
        refinement_instructions.append("Improve fluency and word choice to match the reference text more closely.")
    if quality_scores['rouge'] < 0.5:
        refinement_instructions.append("Ensure all key information from the source is included in the translation.")
    if quality_scores['meteor'] < 0.5:
        refinement_instructions.append("Enhance semantic accuracy and synonym usage to better preserve meaning.")
    
    additional_instructions = " ".join(refinement_instructions) if refinement_instructions else "General refinement for better quality."
    
    user_prompt = (
        f"Source ({source_lang_code}): {source_text}\n"
        f"Initial Translation ({target_lang_code}): {translated_text}\n"
        f"Instructions: {additional_instructions}\n"
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

def find_columns_to_translate(df: pd.DataFrame, source_lang_code: str, api_key: str, sample_size: int = 5) -> List[str]:
    """
    Identify columns containing text in the source language using Deepseek.
    """
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    columns_to_translate = []

    for col in text_columns:
        # Sample non-empty texts from the column
        non_empty_texts = df[col].dropna().astype(str).tolist()
        if not non_empty_texts:
            continue
        
        sample_texts = random.sample(non_empty_texts, min(sample_size, len(non_empty_texts)))
        source_lang_count = 0

        # Detect language of samples
        for text in sample_texts:
            detected_lang = detect_language_with_deepseek(text, api_key)
            if detected_lang == source_lang_code.upper():
                source_lang_count += 1

        # Select column if more than 50% of samples are in source language
        if source_lang_count / len(sample_texts) > 0.5:
            columns_to_translate.append(col)
            logger.info(f"Selected column '{col}' for translation (source language detected in {source_lang_count}/{len(sample_texts)} samples)")

    if not columns_to_translate:
        logger.warning("No columns found with source language texts")
    return columns_to_translate

def read_from_kafka(kafka_config: Dict[str, str], topic: str, batch_size: int = 1000) -> pd.DataFrame:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_config['bootstrap.servers'],
        group_id=kafka_config['group.id'],
        auto_offset_reset=kafka_config['auto.offset.reset'],
        enable_auto_commit=False
    )
    messages = []
    failed_messages = []
    retries = 0
    max_message_retries = 3

    try:
        while len(messages) < batch_size:
            msg_batch = consumer.poll(timeout_ms=1000)
            if not msg_batch:
                continue

            for topic_partition, partition_msgs in msg_batch.items():
                for msg in partition_msgs:
                    try:
                        message_data = msg.value.decode('utf-8')
                        parsed_data = eval(message_data)
                        messages.append(parsed_data)
                        consumer.commit({topic_partition: msg.offset + 1})
                    except Exception as e:
                        logger.warning(f"Failed to parse message {msg.offset}: {str(e)}")
                        failed_messages.append(msg)
            
            if failed_messages and retries < max_message_retries:
                retries += 1
                logger.info(f"Retrying {len(failed_messages)} failed messages (attempt {retries}/{max_message_retries})")
                temp_failed = []
                for failed_msg in failed_messages:
                    try:
                        message_data = failed_msg.value.decode('utf-8')
                        parsed_data = eval(message_data)
                        messages.append(parsed_data)
                        consumer.commit({topic_partition: failed_msg.offset + 1})
                    except Exception as e:
                        logger.warning(f"Retry failed for message {failed_msg.offset}: {str(e)}")
                        temp_failed.append(failed_msg)
                failed_messages = temp_failed

    finally:
        consumer.close()

    if failed_messages:
        logger.error(f"{len(failed_messages)} messages failed after {max_message_retries} retries")
        with open('failed_kafka_messages.pkl', 'wb') as f:
            pickle.dump(failed_messages, f)

    try:
        df = pd.DataFrame(messages)
        return df
    except Exception as e:
        logger.error(f"Failed to create DataFrame from Kafka messages: {str(e)}")
        return pd.DataFrame()

def process_partition(
    partition: pd.DataFrame,
    partition_idx: int,
    translate_columns: List[str],
    source_lang_code: str,
    target_lang_code: str,
    api_key: str,
    additional_instructions: str,
    output_file: str,
    first_partition: bool,
    max_workers: int,
    total_texts: List[int],
    new_translations: List[int],
    quality_scores: List[Dict[str, float]],
    refinements: List[int],
    quality_threshold: float = 0.5,
    quality_check_probability: float = 0.1,
    metric_weights: Dict[str, float] = {'bleu': 0.3, 'rouge': 0.3, 'meteor': 0.4}
):
    logger.info(f"Processing partition {partition_idx + 1} with {len(partition)} rows")
    
    translated_partition = partition.copy()
    translated_partition['thai_text'] = ""

    texts_to_translate = []
    indices = []
    source_texts = []
    for index, row in translated_partition.iterrows():
        for col in translate_columns:
            text = str(row[col])
            if text and text.strip() != "nan":
                texts_to_translate.append((text, source_lang_code, target_lang_code, api_key, additional_instructions))
                indices.append(index)
                source_texts.append(text)
                total_texts[0] += 1
            else:
                translated_partition.at[index, 'thai_text'] = translated_partition.at[index, 'thai_text'] or ""

    if texts_to_translate:
        logger.info(f"Translating {len(texts_to_translate)} texts in partition {partition_idx + 1}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {executor.submit(translate_text, args): args for args in texts_to_translate}
            for future in as_completed(future_to_text):
                try:
                    translated_text = future.result()
                    index = indices[texts_to_translate.index(future_to_text[future])]
                    source_text = source_texts[indices.index(index)]
                    
                    final_translation = translated_text
                    quality_score = None
                    
                    if random.random() < quality_check_probability:
                        reference_text = generate_reference_translation(source_text, source_lang_code, target_lang_code, api_key)
                        if reference_text:
                            quality_score = check_translation_quality(translated_text, reference_text, api_key, metric_weights)
                            quality_scores.append(quality_score)
                            logger.info(f"Quality scores for text at index {index}: BLEU={quality_score['bleu']:.2f}, ROUGE={quality_score['rouge']:.2f}, METEOR={quality_score['meteor']:.2f}, Average={quality_score['average']:.2f}")
                            
                            if quality_score['average'] < quality_threshold:
                                refined_text = refine_translation(source_text, translated_text, source_lang_code, target_lang_code, api_key, quality_score)
                                refined_score = check_translation_quality(refined_text, reference_text, api_key, metric_weights)
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

def translate_text(args: Tuple[str, str, str, str, str]) -> str:
    text, source_lang_code, target_lang_code, api_key, additional_instructions = args
    translated_text = generate_translation_sample(text, source_lang_code, target_lang_code, api_key, additional_instructions)
    return translated_text

def auto_translate_dataset(
    input_file: Optional[str] = None,
    kafka_config: Optional[Dict[str, str]] = None,
    kafka_topic: Optional[str] = None,
    output_file: str = "translated_dataset.csv.gz",
    source_lang_code: str = "EN",
    target_lang_code: str = "TH",
    api_key: str = None,
    additional_instructions: str = "",
    translate_columns: Optional[List[str]] = None,
    max_workers: int = 4,
    n_partitions: int = 4,
    max_retries: int = 3,
    max_failed_partitions: int = 10,
    quality_threshold: float = 0.5,
    kafka_batch_size: int = 1000,
    quality_check_probability: float = 0.1,
    metric_weights: Dict[str, float] = {'bleu': 0.3, 'rouge': 0.3, 'meteor': 0.4}
) -> str:
    """
    Automatically translate dataset from source language to Thai using Dask, Deepseek, Kafka, and BLEU, ROUGE, METEOR metrics.
    """
    start_time = time.time()
    if not api_key:
        api_key = get_deepseek_api_key()

    if not (input_file or (kafka_config and kafka_topic)):
        logger.error("Must provide either input_file or kafka_config with kafka_topic")
        return "Error: Must specify input_file or kafka_config with kafka_topic"

    # Validate metric weights
    if sum(metric_weights.values()) == 0:
        logger.error("Metric weights sum to zero")
        return "Error: Metric weights must sum to more than 0"

    # Remove old temporary files
    temp_output_file = output_file.replace('.csv.gz', '.temp.csv') if output_file.endswith('.csv.gz') else output_file + '.temp.csv'
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
    for f in os.listdir():
        if f.startswith("failed_partition_") and f.endswith(".pkl"):
            os.remove(f)

    # Counters for statistics
    total_texts = [0]
    new_translations = [0]
    quality_scores = []
    refinements = [0]

    # Read dataset
    partitions = []
    if input_file:
        try:
            ddf = dd.read_csv(input_file, blocksize=25e6)
            df_sample = ddf.head(100)  # Sample for column detection
            if translate_columns is None:
                translate_columns = find_columns_to_translate(df_sample, source_lang_code, api_key)
                if not translate_columns:
                    logger.error("No columns selected for translation")
                    return "Error: No columns found to translate"
            else:
                # Validate specified columns
                missing_cols = [col for col in translate_columns if col not in df_sample.columns]
                if missing_cols:
                    logger.error(f"Specified columns not found in dataset: {missing_cols}")
                    return f"Error: Specified columns not found in dataset: {missing_cols}"
                logger.info(f"Using specified columns for translation: {translate_columns}")
            ddf = ddf.repartition(n_partitions=n_partitions)
            logger.info(f"Loaded dataset with {n_partitions} partitions from {input_file}")
            partitions = [(i, ddf.get_partition(i).compute()) for i in range(n_partitions)]
        except Exception as e:
            logger.error(f"Failed to read file {input_file}: {str(e)}")
            return f"Error: Unable to read file {input_file}. Details: {str(e)}"
    elif kafka_config and kafka_topic:
        try:
            df = read_from_kafka(kafka_config, kafka_topic, kafka_batch_size)
            if df.empty:
                logger.error("No data received from Kafka")
                return "Error: No data received from Kafka"
            if translate_columns is None:
                translate_columns = find_columns_to_translate(df, source_lang_code, api_key)
                if not translate_columns:
                    logger.error("No columns selected for translation")
                    return "Error: No columns found to translate"
            else:
                # Validate specified columns
                missing_cols = [col for col in translate_columns if col not in df.columns]
                if missing_cols:
                    logger.error(f"Specified columns not found in dataset: {missing_cols}")
                    return f"Error: Specified columns not found in dataset: {missing_cols}"
                logger.info(f"Using specified columns for translation: {translate_columns}")
            ddf = dd.from_pandas(df, npartitions=n_partitions)
            logger.info(f"Loaded dataset with {n_partitions} partitions from Kafka topic {kafka_topic}")
            partitions = [(i, ddf.get_partition(i).compute()) for i in range(n_partitions)]
        except Exception as e:
            logger.error(f"Failed to read from Kafka: {str(e)}")
            return f"Error: Unable to read from Kafka: {str(e)}"

    # Store failed partitions
    failed_partitions = []
    first_partition = True

    # Process partitions
    for partition_idx, partition in tqdm(partitions, desc="Processing partitions"):
        try:
            if not translate_columns:
                logger.warning(f"No text columns found in partition {partition_idx + 1}")
                continue

            result = process_partition(
                partition, partition_idx, translate_columns, source_lang_code, target_lang_code,
                api_key, additional_instructions, temp_output_file, first_partition,
                max_workers, total_texts, new_translations, quality_scores,
                refinements, quality_threshold, quality_check_probability, metric_weights
            )
            if result:
                failed_partitions.append(result)
                if len(failed_partitions) > max_failed_partitions:
                    logger.error(f"Too many failed partitions: {failed_partitions}")
                    return f"Error: Too many failed partitions: {failed_partitions}"
            first_partition = False
        except Exception as e:
            logger.error(f"Failed to process partition {partition_idx + 1}: {str(e)}")
            failed_partition_file = f"failed_partition_{partition_idx}.pkl"
            with open(failed_partition_file, 'wb') as f:
                pickle.dump(partition, f)
            failed_partitions.append(failed_partition_file)

    # Retry failed partitions
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
                
                result = process_partition(
                    partition, partition_idx, translate_columns, source_lang_code, target_lang_code,
                    api_key, additional_instructions, temp_output_file, False,
                    max_workers, total_texts, new_translations, quality_scores,
                    refinements, quality_threshold, quality_check_probability, metric_weights
                )
                if result:
                    new_failed_partitions.append(result)
                else:
                    os.remove(failed_partition_file)
            except Exception as e:
                logger.error(f"Retry failed for partition {failed_partition_file}: {str(e)}")
                new_failed_partitions.append(failed_partition_file)
        failed_partitions = new_failed_partitions

    # Compress output to gzip
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

    # Generate summary
    elapsed_time = time.time() - start_time
    avg_bleu = sum(s['bleu'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_rouge = sum(s['rouge'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_meteor = sum(s['meteor'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_score = sum(s['average'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
    summary = (
        f"Translation Summary\n"
        f"Status: {'Success' if not failed_partitions else 'Failed'}\n"
        f"Total texts: {total_texts[0]}\n"
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

    if failed_partitions:
        logger.error(f"Failed partitions after {max_retries} retries: {failed_partitions}")
        return f"Error: Some partitions failed: {failed_partitions}"

    logger.info(summary)
    return f"Translation successful! Output saved at {output_file}"

if __name__ == "__main__":
    # Example usage
    input_file = "input_dataset.csv"
    kafka_config = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'translation_group',
        'auto.offset.reset': 'earliest'
    }
    kafka_topic = "translation_topic"
    output_file = "translated_dataset.csv.gz"
    api_key = get_deepseek_api_key()

    # Example 1: Specify columns to translate
    result = auto_translate_dataset(
        input_file=input_file,
        kafka_config=kafka_config,
        kafka_topic=kafka_topic,
        output_file=output_file,
        source_lang_code="EN",
        target_lang_code="TH",
        api_key=api_key,
        translate_columns=["text_en"],
        n_partitions=4,
        max_retries=3,
        max_failed_partitions=10,
        quality_threshold=0.5,
        kafka_batch_size=1000,
        quality_check_probability=0.1,
        metric_weights={'bleu': 0.3, 'rouge': 0.3, 'meteor': 0.4}
    )
    print(result)

    # Example 2: Auto-detect columns
    result = auto_translate_dataset(
        input_file=input_file,
        kafka_config=kafka_config,
        kafka_topic=kafka_topic,
        output_file="translated_dataset_auto.csv.gz",
        source_lang_code="EN",
        target_lang_code="TH",
        api_key=api_key,
        translate_columns=None,
        n_partitions=4,
        max_retries=3,
        max_failed_partitions=10,
        quality_threshold=0.5,
        kafka_batch_size=1000,
        quality_check_probability=0.1,
        metric_weights={'bleu': 0.3, 'rouge': 0.3, 'meteor': 0.4}
    )
    print(result)