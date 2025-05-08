import pandas as pd
import numpy as np
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import thai_words, thai_syllables
import networkx as nx
from sentence_transformers import SentenceTransformer
import json
import yaml
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import warnings
warnings.filterwarnings("ignore")

# โหลดโมเดล sentence transformer สำหรับภาษาไทย (อาจต้องติดตั้งเพิ่มเติม)
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except:
    model = None
    print("Warning: SentenceTransformer model not available. Some functions will be limited.")

# ==================== 1. ตรวจสอบความผิดเพี้ยนของข้อมูล (Hallucination) ====================

def load_factual_reference(ref_path: str = None) -> Dict:
    """โหลดข้อมูลอ้างอิงสำหรับเช็คความถูกต้อง"""
    if ref_path and os.path.exists(ref_path):
        ext = os.path.splitext(ref_path)[1]
        if ext == '.json':
            with open(ref_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(ref_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    # ถ้าไม่มีไฟล์อ้างอิง ใช้ข้อมูลพื้นฐาน
    return {
        "entities": {
            "organizations": ["กระทรวงสาธารณสุข", "องค์การอนามัยโลก", "สหประชาชาติ"],
            "places": ["กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต"],
            "persons": ["นายกรัฐมนตรี", "รัฐมนตรี"]
        },
        "facts": {
            "thai_population": "ประมาณ 70 ล้านคน",
            "covid19_start": "ปลายปี 2019",
            "bangkok_founded": "พ.ศ. 2325" 
        }
    }

def detect_hallucination_markers(df: pd.DataFrame, column: str = 'text', 
                                reference: Dict = None) -> pd.DataFrame:
    """ตรวจหาตัวบ่งชี้ความผิดเพี้ยนของข้อมูล"""
    # คำที่บ่งชี้ความไม่แน่นอนในภาษาไทย
    uncertainty_markers = {
        'ความคลุมเครือ': ['อาจจะ', 'น่าจะ', 'คงจะ', 'บางที', 'อาจ', 'อาจเป็นไปได้'],
        'การคาดเดา': ['คาดว่า', 'เชื่อว่า', 'ดูเหมือนว่า', 'ประมาณ', 'ราวๆ'],
        'ความไม่แน่นอน': ['เกือบจะ', 'แทบจะ', 'มีแนวโน้ม', 'ไม่แน่ใจ', 'สงสัยว่า'],
        'ข้อมูลคลาดเคลื่อน': ['อาจผิด', 'อาจคลาดเคลื่อน', 'ไม่แน่นอน', 'ยังไม่ชัดเจน']
    }

    # คำเชื่อมโยงที่อาจบ่งชี้ hallucination
    logical_inconsistency = [
        'แต่กลับ', 'ทั้งๆที่', 'แม้ว่า', 'ถึงแม้', 'ในขณะที่',
        'แต่ในทางกลับกัน', 'อย่างไรก็ตาม', 'ทว่า'
    ]
    
    # คำที่บ่งชี้ความผิดเพี้ยนแบบเฉพาะเจาะจง
    specific_hallucination = [
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?) ?%', # เปอร์เซ็นต์ที่เฉพาะเจาะจง
        r'พ\.ศ\. ?(\d{4})', # ปี พ.ศ. เฉพาะเจาะจง
        r'ค\.ศ\. ?(\d{4})', # ปี ค.ศ. เฉพาะเจาะจง
        r'(\d+) ?ล้าน', # จำนวนล้าน
        r'(\d+) ?พัน', # จำนวนพัน
        r'(\d+) ?แสน', # จำนวนแสน
    ]
    
    if reference is None:
        reference = load_factual_reference()
    
    # สร้างคอลัมน์ใหม่สำหรับเก็บผลการตรวจสอบ
    df['hallucination_score'] = 0.0
    df['hallucination_markers'] = df[column].apply(lambda x: [])
    
    for idx, row in df.iterrows():
        text = row[column]
        markers = []
        score = 0.0
        
        # ตรวจสอบคำที่บ่งชี้ความไม่แน่นอน
        for marker in uncertainty_markers:
            if marker in text:
                markers.append(f"ความไม่แน่นอน: {marker}")
                score += 0.1
        
        # ตรวจสอบรูปแบบที่บ่งชี้ความผิดเพี้ยนเฉพาะเจาะจง
        for pattern in specific_hallucination:
            matches = re.findall(pattern, text)
            if matches:
                markers.extend([f"ข้อมูลเฉพาะเจาะจง: {m}" for m in matches])
                score += 0.15 * len(matches)
        
        # ตรวจสอบกับข้อมูลอ้างอิง (ถ้ามี)
        if reference and 'entities' in reference:
            # เช็คหน่วยงานที่ไม่มีในรายการอ้างอิง
            tokens = word_tokenize(text)
            for i in range(len(tokens) - 1):
                if tokens[i] == "กระทรวง" and i+1 < len(tokens):
                    entity = f"{tokens[i]}{tokens[i+1]}"
                    if entity not in reference['entities'].get('organizations', []):
                        markers.append(f"องค์กรที่ไม่มีในฐานข้อมูล: {entity}")
                        score += 0.2
        
        # จำกัดคะแนนไว้ที่ 1.0
        score = min(score, 1.0)
        
        df.at[idx, 'hallucination_score'] = score
        df.at[idx, 'hallucination_markers'] = markers
    
    return df

def check_factual_consistency(df: pd.DataFrame, ref_data: Optional[Dict] = None, 
                            column: str = 'text') -> Dict:
    """ตรวจสอบความสอดคล้องกับข้อเท็จจริง"""
    if ref_data is None:
        ref_data = load_factual_reference()
    
    # ตรวจสอบความผิดเพี้ยน
    df = detect_hallucination_markers(df, column, ref_data)
    
    # วิเคราะห์ผลลัพธ์
    hallucination_stats = {
        'average_score': df['hallucination_score'].mean(),
        'high_risk_samples': len(df[df['hallucination_score'] > 0.5]),
        'percentage_with_markers': (len(df[df['hallucination_markers'].str.len() > 0]) / len(df)) * 100
    }
    
    # สร้างกราฟการกระจายคะแนนความผิดเพี้ยน
    plt.figure(figsize=(10, 6))
    sns.histplot(df['hallucination_score'], bins=20)
    plt.title('การกระจายคะแนนความผิดเพี้ยนของข้อมูล')
    plt.xlabel('คะแนนความผิดเพี้ยน')
    plt.ylabel('จำนวนตัวอย่าง')
    plt.tight_layout()
    
    # สร้างเมทริกซ์การประเมิน
    evaluation_matrix = {
        'very_likely_factual': len(df[df['hallucination_score'] <= 0.2]),
        'likely_factual': len(df[(df['hallucination_score'] > 0.2) & (df['hallucination_score'] <= 0.4)]),
        'uncertain': len(df[(df['hallucination_score'] > 0.4) & (df['hallucination_score'] <= 0.6)]),
        'likely_hallucination': len(df[(df['hallucination_score'] > 0.6) & (df['hallucination_score'] <= 0.8)]),
        'very_likely_hallucination': len(df[df['hallucination_score'] > 0.8])
    }
    
    return {
        'hallucination_stats': hallucination_stats,
        'evaluation_matrix': evaluation_matrix,
        'dataframe': df
    }

# ==================== 2. ตรวจสอบอคติและความเป็นกลาง ====================

def load_bias_lexicon() -> Dict[str, List[str]]:
    """โหลดรายการคำที่บ่งชี้อคติประเภทต่างๆ"""
    # ตัวอย่างคำที่อาจแสดงอคติ (ควรพัฒนาและปรับปรุงรายการนี้)
    bias_lexicon = {
        'gender_bias': [
            'แม่บ้าน', 'แม่ครัว', 'ช่างเสริมสวย', 'พยาบาล', 'ผู้หญิงควร', 
            'วิศวกร(?!หญิง)', 'นักวิทยาศาสตร์(?!หญิง)', 'หมอ(?!หญิง)', 'ผู้ชายควร',
            'เพศที่อ่อนแอกว่า', 'เพศที่แข็งแรงกว่า'
        ],
        'racial_bias': [
            'ชาวเขา', 'ไอ้ดำ', 'คนอีสาน', 'ลาว', 'พม่า', 'เขมร', 'แขก', 'ฝรั่ง',
            'ญวน', 'ไทยภูเขา', 'ชนกลุ่มน้อย'
        ],
        'regional_bias': [
            'คนเหนือ(?!มาก)', 'คนอีสาน(?!มาก)', 'คนใต้(?!มาก)', 'บ้านนอก',
            'บ้านป่า', 'บ้านนา', 'ไอ้หนู', 'อีหนู', 'ลูกทุ่ง', 'คนบ้านนอก'
        ],
        'age_bias': [
            'คนแก่', 'คนเฒ่า', 'คนชรา', 'ไอ้แก่', 'เด็กเวร', 'คนรุ่นใหม่ไม่เข้าใจ',
            'คนรุ่นเก่าไม่เข้าใจ', 'เด็กสมัยนี้', 'คนสมัยก่อน', 'วัยทำงาน'
        ],
        'religious_bias': [
            'คนนับถือศาสนา[^.]{1,20}ชอบ', 'คนศาสนา[^.]{1,20}ไม่ชอบ',
            'ศาสนา[^.]{1,20}ดีกว่า', 'ชาว[^.]{1,20}มักจะ', 'คนที่นับถือ[^.]{1,20}มักจะ'
        ],
        'political_bias': [
            'สลิ่ม', 'ควายแดง', 'เผด็จการ', 'ไพร่', 'อำมาตย์', 'นายทุน', 'ทักษิณ', 
            'ฝ่ายค้าน', 'รัฐบาล(?!ไทย)', 'ขวา(?!มือ)', 'ซ้าย(?!มือ)'
        ]
    }
    return bias_lexicon

def check_sentiment_balance(df: pd.DataFrame, column: str = 'text') -> Dict:
    """ตรวจสอบความสมดุลของ sentiment ในข้อความ"""
    # ใช้คำที่แสดงความรู้สึกพื้นฐาน
    positive_words = [
        'ดี', 'เยี่ยม', 'ยอดเยี่ยม', 'สุดยอด', 'ชอบ', 'รัก', 'ประทับใจ', 'พอใจ', 'สวย', 'งาม',
        'เจ๋ง', 'เจ๋งมาก', 'เก่ง', 'สุดเจ๋ง', 'น่ารัก', 'สวยงาม', 'เลิศ', 'ดีเลิศ', 'ไซร้',
        'มีความสุข', 'สุขใจ', 'สบายใจ', 'ดีใจ', 'เบิกบาน', 'สดใส', 'เริงร่า', 'รื่นเริง'
    ]
    
    negative_words = [
        'แย่', 'เลว', 'ไม่ดี', 'แห้ว', 'ผิดหวัง', 'เสียใจ', 'โกรธ', 'เกลียด', 'ไม่ชอบ', 'น่ารำคาญ',
        'ห่วย', 'กาก', 'เน่า', 'แย่มาก', 'เฮงซวย', 'ซวย', 'ไม่เอา', 'เบื่อ', 'รำคาญ', 
        'เศร้า', 'ทุกข์', 'หดหู่', 'กลุ้มใจ', 'อึดอัด', 'หนักใจ', 'วิตก', 'กังวล'
    ]
    
    def calculate_sentiment(text: str) -> float:
        """คำนวณ sentiment score จากจำนวนคำในข้อความ"""
        # นับคำที่แสดงความรู้สึกบวก
        pos_count = sum(1 for word in positive_words if word in text)
        # นับคำที่แสดงความรู้สึกลบ
        neg_count = sum(1 for word in negative_words if word in text)
        
        # คำนวณค่า sentiment ในช่วง -1 ถึง 1
        if pos_count == 0 and neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    # วิเคราะห์ sentiment ของแต่ละข้อความ
    df['sentiment_score'] = df[column].apply(calculate_sentiment)
    
    # วิเคราะห์ sentiment distribution
    sentiment_dist = {
        'positive': len(df[df['sentiment_score'] > 0]),
        'neutral': len(df[df['sentiment_score'] == 0]),
        'negative': len(df[df['sentiment_score'] < 0])
    }
    
    # คำนวณความไม่สมดุล
    total = len(df)
    imbalance = max(abs(sentiment_dist['positive'] - sentiment_dist['negative']) / total, 0)
    
    return {
        'sentiment_distribution': sentiment_dist,
        'sentiment_imbalance': imbalance,
        'average_sentiment': df['sentiment_score'].mean()
    }

def detect_bias(df: pd.DataFrame, column: str = 'text', 
               bias_lexicon: Optional[Dict[str, List[str]]] = None) -> Dict:
    """ตรวจสอบอคติในข้อความ"""
    if bias_lexicon is None:
        bias_lexicon = load_bias_lexicon()
    
    # สร้างคอลัมน์สำหรับเก็บผลการวิเคราะห์
    for bias_type in bias_lexicon.keys():
        df[f'bias_{bias_type}'] = False
    
    df['bias_markers'] = ''
    df['bias_score'] = 0.0
    
    for idx, row in df.iterrows():
        text = row[column]
        markers = []
        bias_types_found = set()
        
        # ตรวจสอบแต่ละประเภทของอคติ
        for bias_type, patterns in bias_lexicon.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    df.at[idx, f'bias_{bias_type}'] = True
                    bias_types_found.add(bias_type)
                    markers.append(f"{bias_type}: {pattern}")
        
        # เก็บผลการวิเคราะห์
        df.at[idx, 'bias_markers'] = '; '.join(markers)
        df.at[idx, 'bias_score'] = len(bias_types_found) / len(bias_lexicon) if bias_lexicon else 0
    
    # สรุปผลการตรวจสอบอคติ
    bias_stats = {bias_type: df[f'bias_{bias_type}'].sum() for bias_type in bias_lexicon.keys()}
    bias_stats['samples_with_any_bias'] = len(df[df['bias_markers'] != ''])
    bias_stats['percentage_with_bias'] = (bias_stats['samples_with_any_bias'] / len(df)) * 100
    
    # ตรวจสอบความสมดุลของ sentiment
    sentiment_analysis = check_sentiment_balance(df, column)
    
    # รวมผลลัพธ์
    return {
        'bias_stats': bias_stats,
        'sentiment_analysis': sentiment_analysis,
        'dataframe': df
    }

def analyze_bias_and_fairness(df: pd.DataFrame, text_column: str = 'text', 
                           plots_dir: Optional[str] = None) -> Dict:
    """วิเคราะห์อคติและความเป็นกลางในชุดข้อมูล"""
    # ตรวจหาอคติ
    bias_results = detect_bias(df, text_column)
    
    # สร้างกราฟแสดงประเภทอคติที่พบ
    bias_lexicon = load_bias_lexicon()
    bias_counts = {bias_type: df[f'bias_{bias_type}'].sum() for bias_type in bias_lexicon.keys()}
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(bias_counts.keys()), y=list(bias_counts.values()))
    plt.title('ประเภทอคติที่พบในชุดข้อมูล')
    plt.xlabel('ประเภทอคติ')
    plt.ylabel('จำนวนตัวอย่าง')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    bias_plot_path = None
    if plots_dir:
        bias_plot_path = os.path.join(plots_dir, 'bias_types.png')
        plt.savefig(bias_plot_path)
        plt.close()
    
    # สร้างกราฟการกระจายของ sentiment
    sentiment_data = bias_results['sentiment_analysis']['sentiment_distribution']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(sentiment_data.keys()), y=list(sentiment_data.values()))
    plt.title('การกระจายของ Sentiment ในชุดข้อมูล')
    plt.xlabel('Sentiment')
    plt.ylabel('จำนวนตัวอย่าง')
    plt.tight_layout()
    
    sentiment_plot_path = None
    if plots_dir:
        sentiment_plot_path = os.path.join(plots_dir, 'sentiment_distribution.png')
        plt.savefig(sentiment_plot_path)
        plt.close()
    
    # สรุปผลลัพธ์
    return {
        'bias_analysis': bias_results['bias_stats'],
        'sentiment_analysis': bias_results['sentiment_analysis'],
        'plots': {
            'bias_types': bias_plot_path,
            'sentiment_distribution': sentiment_plot_path
        },
        'dataframe': bias_results['dataframe']
    }

# ==================== 3. ตรวจสอบความซ้ำซ้อนของข้อมูล ====================

def get_text_similarity(texts: List[str], method: str = 'tfidf') -> npt.NDArray:
    """คำนวณความคล้ายกันระหว่างข้อความ"""
    if method == 'tfidf':
        # ใช้ TF-IDF และ cosine similarity
        vectorizer = TfidfVectorizer(tokenizer=word_tokenize, analyzer='word')
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix).astype(np.float32)
        return similarity_matrix
    elif method == 'embedding' and model is not None:
        # ใช้ sentence embeddings
        embeddings = model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    else:
        # ถ้าไม่มี embedding model หรือเลือกวิธีอื่น
        return get_text_similarity(texts, 'tfidf')

def find_duplicates_and_near_duplicates(df: pd.DataFrame, 
                                      column: str = 'text', 
                                      threshold: float = 0.85,
                                      method: str = 'tfidf') -> Dict:
    """ค้นหาข้อความซ้ำและคล้ายกัน"""
    texts = df[column].tolist()
    similarity_matrix = get_text_similarity(texts, method)
    
    # บันทึกคู่ข้อความที่คล้ายกัน
    similar_pairs = []
    dup_groups = {}
    n = len(texts)
    
    # หาข้อความที่คล้ายกันเกินกว่า threshold
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] >= threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))
                
                # จัดกลุ่มข้อความซ้ำ
                if i not in dup_groups and j not in dup_groups:
                    # สร้างกลุ่มใหม่
                    group_id = len(dup_groups) + 1
                    dup_groups[i] = group_id
                    dup_groups[j] = group_id
                elif i in dup_groups and j not in dup_groups:
                    # เพิ่ม j เข้ากลุ่มของ i
                    dup_groups[j] = dup_groups[i]
                elif i not in dup_groups and j in dup_groups:
                    # เพิ่ม i เข้ากลุ่มของ j
                    dup_groups[i] = dup_groups[j]
                # ถ้าทั้ง i และ j อยู่ในกลุ่มอยู่แล้ว ไม่ต้องทำอะไร
    
    # จัดกลุ่มข้อความซ้ำ
    duplicate_groups = {}
    for idx, group_id in dup_groups.items():
        if group_id not in duplicate_groups:
            duplicate_groups[group_id] = []
        duplicate_groups[group_id].append(idx)
    
    # เตรียมเก็บข้อมูลความคล้ายสำหรับแต่ละตัวอย่าง
    df['duplication_score'] = 0.0
    df['similar_samples'] = df.index.map(lambda x: [])
    
    # คำนวณความคล้ายเฉลี่ยกับตัวอย่างอื่นๆ
    for i in range(n):
        similar_indices = []
        for j in range(n):
            if i != j and similarity_matrix[i, j] >= threshold:
                similar_indices.append(j)
        
        if similar_indices:
            df.at[i, 'duplication_score'] = np.mean([similarity_matrix[i, j] for j in similar_indices])
            df.at[i, 'similar_samples'] = similar_indices
    
    # นับจำนวนข้อความซ้ำและคล้ายคลึง
    exact_duplicates = len([g for g in duplicate_groups.values() if len(g) > 1])
    near_duplicates = len(similar_pairs) - exact_duplicates
    
    # คำนวณความหลากหลายของข้อมูล
    diversity_score = 1.0 - (len(similar_pairs) / (n * (n-1) / 2) if n > 1 else 0)
    
    # สร้างกราฟเครือข่ายความคล้ายคลึง
    if len(similar_pairs) > 0:
        G = nx.Graph()
        for i, j, sim in similar_pairs:
            G.add_edge(i, j, weight=sim)
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.3)
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)
        # วาด edges ทีละเส้น
        for (u, v), weight in zip(edges, weights):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=float(weight), alpha=0.5)
        plt.title('เครือข่ายความคล้ายคลึงของข้อความ')
        plt.axis('off')
    
    # สรุปผลการวิเคราะห์
    duplication_stats = {
        'exact_duplicates': exact_duplicates,
        'near_duplicates': near_duplicates,
        'total_samples': n,
        'diversity_score': diversity_score,
        'duplicate_groups': len(duplicate_groups),
        'average_duplication_score': df['duplication_score'].mean()
    }
    
    return {
        'duplication_stats': duplication_stats,
        'duplicate_groups': duplicate_groups,
        'similarity_pairs': similar_pairs,
        'dataframe': df
    }

def analyze_duplicates(df: pd.DataFrame, text_column: str = 'text', 
                    plots_dir: Optional[str] = None,
                    threshold: float = 0.85) -> Dict:
    """วิเคราะห์ความซ้ำซ้อนในชุดข้อมูล"""
    # ตรวจสอบข้อมูลซ้ำ
    duplicate_results = find_duplicates_and_near_duplicates(df, text_column, threshold)
    
    # สร้างกราฟแสดงการกระจายคะแนนความซ้ำซ้อน
    plt.figure(figsize=(10, 6))
    sns.histplot(duplicate_results['dataframe']['duplication_score'], bins=20)
    plt.title('การกระจายคะแนนความซ้ำซ้อน')
    plt.xlabel('คะแนนความซ้ำซ้อน')
    plt.ylabel('จำนวนตัวอย่าง')
    plt.tight_layout()
    
    dup_score_plot_path = None
    if plots_dir:
        dup_score_plot_path = os.path.join(plots_dir, 'duplication_score.png')
        plt.savefig(dup_score_plot_path)
        plt.close()
    
    # สร้างกราฟแสดงขนาดกลุ่มข้อความซ้ำ
    if duplicate_results['duplicate_groups']:
        group_sizes = [len(group) for group in duplicate_results['duplicate_groups'].values()]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(group_sizes, bins=min(20, len(set(group_sizes))))
        plt.title('ขนาดของกลุ่มข้อความซ้ำ')
        plt.xlabel('จำนวนข้อความในกลุ่ม')
        plt.ylabel('จำนวนกลุ่ม')
        plt.tight_layout()
        
        group_size_plot_path = None
        if plots_dir:
            group_size_plot_path = os.path.join(plots_dir, 'duplicate_group_sizes.png')
            plt.savefig(group_size_plot_path)
            plt.close()
    else:
        group_size_plot_path = None
    
    # สรุปผลลัพธ์
    return {
        'duplication_analysis': duplicate_results['duplication_stats'],
        'plots': {
            'duplication_score': dup_score_plot_path,
            'duplicate_group_sizes': group_size_plot_path
        },
        'dataframe': duplicate_results['dataframe']
    }

# ==================== ฟังก์ชันหลักสำหรับการทดสอบคุณภาพข้อมูลขั้นสูง ====================

def run_advanced_quality_check(df: pd.DataFrame, text_column: str = 'text', 
                             output_dir: str = 'analysis_output', 
                             ref_data: Optional[Dict] = None,
                             similarity_threshold: float = 0.85) -> Dict:
    """ตรวจสอบคุณภาพข้อมูลขั้นสูง"""
    print("\nกำลังตรวจสอบคุณภาพข้อมูลขั้นสูง...")
    print("=" * 50)
    
    # ตรวจสอบและสร้างโครงสร้างไดเรกทอรี
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(output_dir, f"advanced_analysis_{timestamp}")
    plots_dir = os.path.join(output_subdir, 'plots')
    reports_dir = os.path.join(output_subdir, 'reports')
    stats_dir = os.path.join(output_subdir, 'stats')
    
    for directory in [output_subdir, plots_dir, reports_dir, stats_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 1. ตรวจสอบความผิดเพี้ยนของข้อมูล
    print("\n1. ตรวจสอบความผิดเพี้ยนของข้อมูล...")
    hallucination_results = check_factual_consistency(df.copy(), ref_data, text_column)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(hallucination_results['dataframe']['hallucination_score'], bins=20)
    plt.title('การกระจายคะแนนความผิดเพี้ยนของข้อมูล')
    plt.xlabel('คะแนนความผิดเพี้ยน')
    plt.ylabel('จำนวนตัวอย่าง')
    plt.tight_layout()
    hallucination_plot_path = os.path.join(plots_dir, 'hallucination_score.png')
    plt.savefig(hallucination_plot_path)
    plt.close()
    
    # 2. ตรวจสอบอคติและความเป็นกลาง
    print("\n2. ตรวจสอบอคติและความเป็นกลาง...")
    bias_results = analyze_bias_and_fairness(df.copy(), text_column, plots_dir)
    
    # 3. ตรวจสอบความซ้ำซ้อนของข้อมูล
    print("\n3. ตรวจสอบความซ้ำซ้อนของข้อมูล...")
    duplicate_results = analyze_duplicates(df.copy(), text_column, plots_dir, similarity_threshold)
    
    # รวมผลลัพธ์
    results = {
        'dataset_info': {
            'filename': 'ชุดข้อมูลที่วิเคราะห์',
            'total_samples': len(df),
            'timestamp': timestamp
        },
        'hallucination_analysis': {
            'stats': hallucination_results['hallucination_stats'],
            'evaluation_matrix': hallucination_results['evaluation_matrix'],
            'plot': hallucination_plot_path
        },
        'bias_analysis': {
            'stats': bias_results['bias_analysis'],
            'sentiment': bias_results['sentiment_analysis'],
            'plots': bias_results['plots']
        },
        'duplication_analysis': {
            'stats': duplicate_results['duplication_analysis'],
            'plots': duplicate_results['plots']
        }
    }
    
    # บันทึกผลลัพธ์
    with open(os.path.join(stats_dir, 'advanced_analysis_results.json'), 'w', encoding='utf-8') as f:
        # Convert non-serializable objects to strings
        results_serializable = json.dumps(results, default=lambda o: str(o) if not isinstance(o, (dict, list, str, int, float, bool, type(None))) else o, ensure_ascii=False, indent=2)
        f.write(results_serializable)
    
    # สร้างรายงานสรุป
    report = ["รายงานการวิเคราะห์คุณภาพข้อมูลขั้นสูง", "=" * 40 + "\n"]
    report.append(f"จำนวนตัวอย่างทั้งหมด: {len(df)}")
    report.append(f"วันและเวลาที่วิเคราะห์: {timestamp}\n")
    
    report.append("\n1. การตรวจสอบความผิดเพี้ยนของข้อมูล")
    report.append("-----------------------------------")
    report.append(f"คะแนนความผิดเพี้ยนเฉลี่ย: {hallucination_results['hallucination_stats']['average_score']:.4f}")
    report.append(f"ตัวอย่างที่มีความเสี่ยงสูง: {hallucination_results['hallucination_stats']['high_risk_samples']} ตัวอย่าง")
    report.append(f"เปอร์เซ็นต์ข้อมูลที่มีตัวบ่งชี้ความผิดเพี้ยน: {hallucination_results['hallucination_stats']['percentage_with_markers']:.2f}%\n")
    
    report.append("\n2. การตรวจสอบอคติและความเป็นกลาง")
    report.append("----------------------------------")
    report.append(f"จำนวนตัวอย่างที่พบอคติ: {bias_results['bias_analysis']['samples_with_any_bias']} ตัวอย่าง")
    report.append(f"เปอร์เซ็นต์ข้อมูลที่มีอคติ: {bias_results['bias_analysis']['percentage_with_bias']:.2f}%")
    report.append(f"ความไม่สมดุลของ Sentiment: {bias_results['sentiment_analysis']['sentiment_imbalance']:.4f}")
    report.append(f"ค่าเฉลี่ย Sentiment: {bias_results['sentiment_analysis']['average_sentiment']:.4f}\n")
    
    report.append("\n3. การตรวจสอบความซ้ำซ้อนของข้อมูล")
    report.append("----------------------------------")
    report.append(f"จำนวนข้อความซ้ำซ้อน: {duplicate_results['duplication_analysis']['exact_duplicates']} คู่")
    report.append(f"จำนวนข้อความคล้ายคลึง: {duplicate_results['duplication_analysis']['near_duplicates']} คู่")
    report.append(f"คะแนนความหลากหลาย: {duplicate_results['duplication_analysis']['diversity_score']:.4f}")
    report.append(f"จำนวนกลุ่มข้อความซ้ำ: {duplicate_results['duplication_analysis']['duplicate_groups']}")
    
    with open(os.path.join(reports_dir, 'advanced_quality_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\nการวิเคราะห์คุณภาพข้อมูลขั้นสูงเสร็จสมบูรณ์")
    print(f"ผลลัพธ์ถูกบันทึกไว้ที่: {output_subdir}")
    
    return {
        'results': results,
        'output_dir': output_subdir,
        'report_path': os.path.join(reports_dir, 'advanced_quality_report.txt')
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced dataset quality checker')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                      help='Base directory for analysis outputs')
    parser.add_argument('--text_column', type=str, default='text',
                      help='Column name containing text data')
    parser.add_argument('--reference_file', type=str, default=None,
                      help='Path to reference data file (JSON or YAML) for fact checking')
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                      help='Threshold for text similarity (0.0-1.0)')
    
    args = parser.parse_args()
    
    # โหลดข้อมูล
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
    else:
        df = pd.read_csv(args.input_file)
        
        # โหลดข้อมูลอ้างอิง (ถ้ามี)
        ref_data = None
        if args.reference_file:
            ref_data = load_factual_reference(args.reference_file)
        
        # ตรวจสอบคุณภาพข้อมูลขั้นสูง
        results = run_advanced_quality_check(
            df, 
            text_column=args.text_column, 
            output_dir=args.output_dir,
            ref_data=ref_data,
            similarity_threshold=args.similarity_threshold
        )
