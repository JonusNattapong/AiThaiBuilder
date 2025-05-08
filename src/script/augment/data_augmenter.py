import os
import re
import random
import argparse
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

try:
    from googletrans import Translator
    translation_available = True
except ImportError:
    translation_available = False

class ThaiDataAugmenter:
    def __init__(self, data: str, text_column: str = 'text', label_column: str = None, 
                 output_dir: str = 'augmented_data', random_state: int = 42):
        self.data = self._load_data(data)
        self.text_column = text_column
        self.label_column = label_column
        self.output_dir = output_dir
        self.random_state = random_state
        self.tokenizer = word_tokenize
        self.stopwords = set(thai_stopwords())
        random.seed(random_state)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _load_data(self, data: str) -> pd.DataFrame:
        """โหลดข้อมูลจากไฟล์"""
        if data.endswith('.csv'):
            return pd.read_csv(data)
        elif data.endswith('.tsv'):
            return pd.read_csv(data, sep='\t')
        elif data.endswith('.json'):
            return pd.read_json(data)
        elif data.endswith('.jsonl'):
            return pd.read_json(data, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {data}")
    
    def _tokenize_for_augment(self, text: str) -> List[str]:
        """แบ่งคำสำหรับการ augment โดยตัด stopwords ออก"""
        # Normalize ข้อความก่อน tokenize
        normalized_text = normalize(text)
        # แบ่งคำ
        tokens = self.tokenizer(normalized_text)
        # กรองเฉพาะคำที่ไม่ใช่ stopwords และไม่ใช่เครื่องหมายวรรคตอน
        filtered_tokens = [token for token in tokens if token not in self.stopwords and not re.match(r'^[!@#$%^&*(),.;:\'"\/\[\]{}|<>]+$', token)]
        return filtered_tokens
    
    def random_deletion(self, text: str, p: float = 0.15) -> str:
        """ลบคำสุ่มจากข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            p: ความน่าจะเป็นของการลบแต่ละคำ (0-1)
            
        Returns:
            str: ข้อความที่ถูกลบคำสุ่ม
        """
        tokens = self.tokenizer(text)
        if len(tokens) <= 1:
            return text
        
        # เก็บคำที่ไม่ถูกลบ
        keep_tokens = []
        for token in tokens:
            # คำสั้นหรือคำไทยสำคัญที่ควรเก็บไว้
            if len(token) <= 1 or random.random() > p:
                keep_tokens.append(token)
        
        # หากลบหมดให้คืนข้อความเดิม
        if len(keep_tokens) == 0:
            keep_tokens = tokens
        
        return ''.join(keep_tokens)
    
    def random_swap(self, text: str, n: int = 3) -> str:
        """สลับตำแหน่งคำสุ่มในข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            n: จำนวนครั้งที่สลับคำ
            
        Returns:
            str: ข้อความที่ถูกสลับคำ
        """
        tokens = self.tokenizer(text)
        if len(tokens) <= 1:
            return text
        
        # ทำสำเนา tokens
        new_tokens = tokens.copy()
        
        for _ in range(min(n, len(tokens))):
            # สุ่มเลือกตำแหน่งสองตำแหน่งและสลับกัน
            index1, index2 = random.sample(range(len(new_tokens)), 2)
            new_tokens[index1], new_tokens[index2] = new_tokens[index2], new_tokens[index1]
        
        return ''.join(new_tokens)
    
    def random_insertion(self, text: str, n: int = 3) -> str:
        """แทรกคำสุ่มลงในข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            n: จำนวนคำที่แทรก
            
        Returns:
            str: ข้อความที่ถูกแทรกคำเพิ่ม
        """
        tokens = self.tokenizer(text)
        if len(tokens) <= 1:
            return text
        
        # ทำสำเนา tokens
        new_tokens = tokens.copy()
        
        for _ in range(n):
            # เลือกคำสุ่มจากข้อความเดิม
            insert_token = tokens[random.randint(0, len(tokens) - 1)]
            
            # เลือกตำแหน่งสุ่มสำหรับแทรก
            insert_pos = random.randint(0, len(new_tokens))
            new_tokens.insert(insert_pos, insert_token)
        
        return ''.join(new_tokens)
    
    def synonym_replacement(self, text: str, n: int = 3) -> str:
        """แทนที่คำด้วยคำที่มีความหมายใกล้เคียง
        
        Args:
            text: ข้อความต้นฉบับ
            n: จำนวนคำที่ต้องการแทนที่
            
        Returns:
            str: ข้อความที่ถูกแทนที่ด้วยคำใกล้เคียง
        """
        try:
            # ใช้ nlpaug สำหรับการแทนที่ด้วย synonym
            augmenter = naw.SynonymAug(
                aug_src='wordnet',
                aug_max=n
            )
            augmented_text = augmenter.augment(text)
            return augmented_text
        except Exception as e:
            print(f"Synonym replacement failed: {str(e)}")
            return text
    
    def word_embedding_replacement(self, text: str, n: int = 3) -> str:
        """แทนที่คำด้วยคำที่มี embedding ใกล้เคียง
        
        Args:
            text: ข้อความต้นฉบับ
            n: จำนวนคำที่ต้องการแทนที่
            
        Returns:
            str: ข้อความที่ถูกแทนที่ด้วยคำที่มี embedding ใกล้เคียง
        """
        try:
            # ใช้ nlpaug สำหรับ word embedding replacement
            augmenter = naw.WordEmbsAug(
                model_type='word2vec',
                model_path=None,  # ใช้โมเดลภาษาไทยที่มีอยู่ใน nlpaug
                aug_max=n,
                aug_p=0.3,
                action="substitute"
            )
            augmented_text = augmenter.augment(text)
            return augmented_text
        except Exception as e:
            print(f"Word embedding replacement failed: {str(e)}")
            return text
    
    def back_translation(self, text: str) -> str:
        """แปลงข้อความไปภาษาอื่นแล้วแปลกลับมาเป็นภาษาไทย
        
        Args:
            text: ข้อความต้นฉบับภาษาไทย
            
        Returns:
            str: ข้อความที่ผ่านการแปลภาษาไปกลับ
        """
        if not translation_available:
            return text
        
        try:
            # แปลจากไทยเป็นอังกฤษ
            english_text = translate(text, source='th', target='en')
            # แปลจากอังกฤษกลับเป็นไทย
            back_th_text = translate(english_text, source='en', target='th')
            return back_th_text
        except Exception as e:
            print(f"Back translation failed: {str(e)}")
            return text
    
    def character_replacement(self, text: str, p: float = 0.1) -> str:
        """แทนที่ตัวอักษรบางตัวในข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            p: ความน่าจะเป็นของการแทนที่แต่ละตัวอักษร (0-1)
            
        Returns:
            str: ข้อความที่ถูกแทนที่ตัวอักษร
        """
        try:
            # ใช้ nlpaug สำหรับ character replacement
            augmenter = nac.CharacterOcrAug(
                aug_char_p=p, 
                aug_word_p=p*2
            )
            augmented_text = augmenter.augment(text)
            return augmented_text
        except Exception as e:
            print(f"Character replacement failed: {str(e)}")
            return text
    
    def contextual_augmentation(self, text: str, p: float = 0.15) -> str:
        """ใช้โมเดล BERT ในการทำ contextual augmentation
        
        Args:
            text: ข้อความต้นฉบับ
            p: ความน่าจะเป็นของการแทนที่แต่ละคำ (0-1)
            
        Returns:
            str: ข้อความที่ถูก augment ด้วย contextual model
        """
        try:
            # ใช้ nlpaug สำหรับ contextual augmentation
            augmenter = naw.ContextualWordEmbsAug(
                model_path='wangchanberta-base-att-spm-uncased',
                action="substitute",
                aug_p=p
            )
            augmented_text = augmenter.augment(text)
            return augmented_text
        except Exception as e:
            print(f"Contextual augmentation failed: {str(e)}")
            return text
    
    def sentence_augmentation(self, text: str, n: int = 1) -> str:
        """เพิ่มหรือลดประโยคในข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            n: จำนวนประโยคที่ต้องการเพิ่มหรือลด
            
        Returns:
            str: ข้อความที่ถูก augment ในระดับประโยค
        """
        try:
            # ตัดเป็นประโยคก่อน
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                return text
            
            # สุ่มเลือกว่าจะลบหรือสลับประโยค
            op = random.choice(['delete', 'swap'])
            
            if op == 'delete' and len(sentences) > n:
                # ลบประโยคสุ่ม
                for _ in range(min(n, len(sentences) - 1)):
                    del_idx = random.randint(0, len(sentences) - 1)
                    sentences.pop(del_idx)
            
            elif op == 'swap':
                # สลับประโยคสุ่ม
                for _ in range(min(n, len(sentences))):
                    idx1, idx2 = random.sample(range(len(sentences)), 2)
                    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
            
            # รวมประโยคกลับเป็นข้อความ
            return '. '.join(sentences) + '.'
        
        except Exception as e:
            print(f"Sentence augmentation failed: {str(e)}")
            return text
    
    def easy_data_augmentation(self, text: str, alpha_sr: float = 0.1, alpha_ri: float = 0.1, 
                               alpha_rs: float = 0.1, alpha_rd: float = 0.1, num_aug: int = 4) -> List[str]:
        """ใช้เทคนิค Easy Data Augmentation รวม 4 เทคนิค
        
        Args:
            text: ข้อความต้นฉบับ
            alpha_sr: ความน่าจะเป็นของ synonym replacement
            alpha_ri: ความน่าจะเป็นของ random insertion
            alpha_rs: ความน่าจะเป็นของ random swap
            alpha_rd: ความน่าจะเป็นของ random deletion
            num_aug: จำนวนข้อความที่ต้องการสร้าง
            
        Returns:
            List[str]: รายการข้อความที่ถูก augment
        """
        tokens = self.tokenizer(text)
        words = len(tokens)
        
        augmented_texts = []
        num_new_per_technique = int(num_aug / 4) + 1
        
        # SR: Synonym Replacement
        n_sr = max(1, int(alpha_sr * words))
        for _ in range(num_new_per_technique):
            augmented_texts.append(self.synonym_replacement(text, n=n_sr))
        
        # RI: Random Insertion
        n_ri = max(1, int(alpha_ri * words))
        for _ in range(num_new_per_technique):
            augmented_texts.append(self.random_insertion(text, n=n_ri))
        
        # RS: Random Swap
        n_rs = max(1, int(alpha_rs * words))
        for _ in range(num_new_per_technique):
            augmented_texts.append(self.random_swap(text, n=n_rs))
        
        # RD: Random Deletion
        for _ in range(num_new_per_technique):
            augmented_texts.append(self.random_deletion(text, p=alpha_rd))
        
        # ตรวจสอบให้แน่ใจว่าได้จำนวนตามที่ต้องการ
        augmented_texts = augmented_texts[:num_aug]
        return augmented_texts
    
    def augment_text(self, text: str, techniques: List[str] = None, num_per_technique: int = 1) -> List[Dict[str, Any]]:
        """ใช้เทคนิคหลายอย่างในการ augment ข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            techniques: รายการเทคนิคที่ต้องการใช้ (หากไม่ระบุจะใช้ทั้งหมด)
            num_per_technique: จำนวนตัวอย่างที่ต้องการสร้างต่อเทคนิค
            
        Returns:
            List[Dict]: รายการข้อความที่ augment พร้อมเมตาดาต้า
        """
        available_techniques = {
            'random_deletion': self.random_deletion,
            'random_swap': self.random_swap,
            'random_insertion': self.random_insertion,
            'synonym_replacement': self.synonym_replacement,
            'word_embedding': self.word_embedding_replacement,
            'back_translation': self.back_translation,
            'character_replacement': self.character_replacement,
            'contextual_augmentation': self.contextual_augmentation,
            'sentence_augmentation': self.sentence_augmentation
        }
        
        if techniques is None:
            # ใช้ทุกเทคนิคยกเว้น back_translation ถ้าไม่มีโมเดลแปลภาษา
            techniques = list(available_techniques.keys())
            if not translation_available and 'back_translation' in techniques:
                techniques.remove('back_translation')
        
        augmented_results = []
        
        # เก็บข้อความต้นฉบับไว้ด้วย
        augmented_results.append({
            'text': text,
            'technique': 'original',
            'original': text
        })
        
        # ทำการ augment ด้วยแต่ละเทคนิค
        for technique in techniques:
            if technique not in available_techniques:
                print(f"Warning: Unknown technique '{technique}'. Skipping.")
                continue
            
            aug_func = available_techniques[technique]
            
            for _ in range(num_per_technique):
                try:
                    augmented_text = aug_func(text)
                    # เก็บเฉพาะข้อความที่เปลี่ยนแปลงจากต้นฉบับ
                    if augmented_text != text:
                        augmented_results.append({
                            'text': augmented_text,
                            'technique': technique,
                            'original': text
                        })
                except Exception as e:
                    print(f"Error applying {technique}: {str(e)}")
        
        return augmented_results
    
    def augment_dataset(self, techniques: List[str] = None, num_per_technique: int = 1, 
                        balance_labels: bool = False, min_length: int = 10) -> pd.DataFrame:
        """สร้างชุดข้อมูลใหม่จากชุดข้อมูลเดิมโดยใช้เทคนิค augmentation
        
        Args:
            techniques: รายการเทคนิคที่ต้องการใช้
            num_per_technique: จำนวนตัวอย่างที่ต้องการสร้างต่อเทคนิคต่อตัวอย่าง
            balance_labels: ทำให้จำนวนตัวอย่างในแต่ละ label มีจำนวนเท่ากันหรือไม่
            min_length: ความยาวขั้นต่ำของข้อความที่ต้องการ augment
            
        Returns:
            pd.DataFrame: ชุดข้อมูลใหม่ที่รวมข้อความ augment ด้วย
        """
        augmented_data = []
        
        # กรณีที่มี label และต้องการทำให้ balanced
        if self.label_column and balance_labels:
            # นับจำนวนตัวอย่างในแต่ละ label
            label_counts = self.data[self.label_column].value_counts()
            max_count = label_counts.max()
            
            # คำนวณจำนวนตัวอย่างที่ต้องสร้างเพิ่มในแต่ละ label
            for label, count in label_counts.items():
                samples_to_generate = max_count - count
                if samples_to_generate <= 0:
                    continue
                
                # ดึงข้อมูลของ label นี้
                label_data = self.data[self.data[self.label_column] == label]
                
                # คำนวณจำนวนตัวอย่างที่ต้อง augment ต่อข้อความ
                samples_per_text = max(1, samples_to_generate // len(label_data))
                
                # สร้างข้อมูลเพิ่ม
                for _, row in tqdm(label_data.iterrows(), total=len(label_data), desc=f"Augmenting label '{label}'"):
                    text = row[self.text_column]
                    
                    # ข้ามข้อความที่สั้นเกินไป
                    if len(text) < min_length:
                        continue
                    
                    # สร้างข้อความใหม่
                    augmented_texts = self.augment_text(
                        text, 
                        techniques=techniques, 
                        num_per_technique=min(samples_per_text, num_per_technique)
                    )
                    
                    # เลือกเฉพาะข้อความที่ต้องการ (ไม่รวมต้นฉบับ)
                    augmented_texts = [t for t in augmented_texts if t['technique'] != 'original']
                    augmented_texts = augmented_texts[:samples_per_text]
                    
                    # สร้าง rows ใหม่
                    for aug_item in augmented_texts:
                        new_row = row.to_dict()
                        new_row[self.text_column] = aug_item['text']
                        new_row['augmentation_technique'] = aug_item['technique']
                        new_row['original_text'] = aug_item['original']
                        augmented_data.append(new_row)
        
        # กรณีที่ไม่ต้องการ balance labels
        else:
            # Loop ผ่านทุกแถวในชุดข้อมูล
            for i, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Augmenting dataset"):
                text = row[self.text_column]
                
                # ข้ามข้อความที่สั้นเกินไป
                if len(text) < min_length:
                    continue
                
                # สร้างข้อความใหม่
                augmented_texts = self.augment_text(
                    text, 
                    techniques=techniques, 
                    num_per_technique=num_per_technique
                )
                
                # สร้าง rows ใหม่
                for aug_item in augmented_texts:
                    new_row = row.to_dict()
                    new_row[self.text_column] = aug_item['text']
                    new_row['augmentation_technique'] = aug_item['technique']
                    new_row['original_text'] = aug_item['original'] if aug_item['technique'] != 'original' else None
                    augmented_data.append(new_row)
        
        # สร้าง DataFrame ใหม่
        augmented_df = pd.DataFrame(augmented_data)
        
        return augmented_df
    
    def save_augmented_data(self, augmented_df: pd.DataFrame, output_filename: str = None, 
                            include_original: bool = True, format: str = 'csv') -> str:
        """บันทึกข้อมูลที่ augment แล้วลงไฟล์
        
        Args:
            augmented_df: DataFrame ที่มีข้อมูลที่ augment แล้ว
            output_filename: ชื่อไฟล์สำหรับบันทึก (หากไม่ระบุจะสร้างชื่ออัตโนมัติ)
            include_original: รวมข้อมูลต้นฉบับในไฟล์ที่บันทึกหรือไม่
            format: รูปแบบไฟล์ ('csv', 'tsv', 'json', 'jsonl')
            
        Returns:
            str: path ของไฟล์ที่บันทึก
        """
        # สร้างชื่อไฟล์ถ้าไม่ได้ระบุ
        if output_filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"augmented_data_{timestamp}"
        
        # สร้าง path เต็ม
        if not output_filename.endswith(f'.{format}'):
            output_filename = f"{output_filename}.{format}"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # กรองข้อมูลตามที่ต้องการ
        if not include_original:
            augmented_df = augmented_df[augmented_df['augmentation_technique'] != 'original']
        
        # บันทึกไฟล์ตามรูปแบบที่ต้องการ
        if format == 'csv':
            augmented_df.to_csv(output_path, index=False, encoding='utf-8')
        elif format == 'tsv':
            augmented_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        elif format == 'json':
            augmented_df.to_json(output_path, orient='records', force_ascii=False, indent=4)
        elif format == 'jsonl':
            augmented_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Augmented data saved to: {output_path}")
        return output_path
    
    def augment_and_save(self, techniques: List[str] = None, num_per_technique: int = 1,
                         balance_labels: bool = False, min_length: int = 10,
                         output_filename: str = None, include_original: bool = True,
                         format: str = 'csv') -> str:
        """Augment ข้อมูลและบันทึกในคำสั่งเดียว
        
        Args:
            techniques: รายการเทคนิคที่ต้องการใช้
            num_per_technique: จำนวนตัวอย่างที่ต้องการสร้างต่อเทคนิคต่อตัวอย่าง
            balance_labels: ทำให้จำนวนตัวอย่างในแต่ละ label มีจำนวนเท่ากันหรือไม่
            min_length: ความยาวขั้นต่ำของข้อความที่ต้องการ augment
            output_filename: ชื่อไฟล์สำหรับบันทึก
            include_original: รวมข้อมูลต้นฉบับในไฟล์ที่บันทึกหรือไม่
            format: รูปแบบไฟล์ ('csv', 'tsv', 'json', 'jsonl')
            
        Returns:
            str: path ของไฟล์ที่บันทึก
        """
        augmented_df = self.augment_dataset(
            techniques=techniques,
            num_per_technique=num_per_technique,
            balance_labels=balance_labels,
            min_length=min_length
        )
        
        return self.save_augmented_data(
            augmented_df=augmented_df,
            output_filename=output_filename,
            include_original=include_original,
            format=format
        )

def main():
    parser = argparse.ArgumentParser(description='Thai Data Augmentation')
    parser.add_argument('--input_file', type=str, required=True, 
                      help='Path to input file (CSV, TSV, JSON, or JSONL)')
    parser.add_argument('--output_dir', type=str, default='augmented_data',
                      help='Directory to save augmented data')
    parser.add_argument('--text_column', type=str, default='text',
                      help='Column name containing text data')
    parser.add_argument('--label_column', type=str, default=None,
                      help='Column name containing labels (optional)')
    parser.add_argument('--techniques', type=str, nargs='+', 
                       choices=['random_deletion', 'random_swap', 'random_insertion', 
                                'synonym_replacement', 'word_embedding', 'back_translation',
                                'character_replacement', 'contextual_augmentation', 
                                'sentence_augmentation', 'all'],
                       default=['all'],
                       help='Augmentation techniques to use')
    parser.add_argument('--num_per_technique', type=int, default=1,
                      help='Number of augmented examples to generate per technique')
    parser.add_argument('--balance_labels', action='store_true',
                      help='Balance the number of examples per label')
    parser.add_argument('--min_length', type=int, default=10,
                      help='Minimum text length to augment')
    parser.add_argument('--output_format', type=str, choices=['csv', 'tsv', 'json', 'jsonl'],
                      default='csv', help='Output file format')
    parser.add_argument('--include_original', action='store_true', default=True,
                      help='Include original text in output')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # แปลง 'all' เป็นรายการเทคนิคทั้งหมด
    if 'all' in args.techniques:
        args.techniques = ['random_deletion', 'random_swap', 'random_insertion', 
                         'synonym_replacement', 'word_embedding', 
                         'character_replacement', 'contextual_augmentation', 
                         'sentence_augmentation']
        # เพิ่ม back_translation ถ้ามีโมเดลแปลภาษา
        if translation_available:
            args.techniques.append('back_translation')
    
    # สร้าง augmenter
    augmenter = ThaiDataAugmenter(
        data=args.input_file,
        text_column=args.text_column,
        label_column=args.label_column,
        output_dir=args.output_dir,
        random_state=args.random_seed
    )
    
    # Augment และบันทึกข้อมูล
    output_path = augmenter.augment_and_save(
        techniques=args.techniques,
        num_per_technique=args.num_per_technique,
        balance_labels=args.balance_labels,
        min_length=args.min_length,
        include_original=args.include_original,
        format=args.output_format
    )
    
    print(f"Data augmentation completed. Output saved to {output_path}")

if __name__ == "__main__":
    main()