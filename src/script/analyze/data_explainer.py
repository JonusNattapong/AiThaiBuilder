import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from wordcloud import WordCloud
from tqdm import tqdm
from pythainlp import word_tokenize, pos_tag
from pythainlp.util import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

class DataExplainer:
    def __init__(self, data: str, text_column: str = 'text', label_column: Optional[str] = None, output_dir: str = 'explainer_output'):
        self.data = pd.read_csv(data)
        self.text_column = text_column
        self.label_column = label_column
        self.output_dir = output_dir
        self.stopwords = set(word_tokenize('และ หรือ แต่ เพราะ เพื่อ ถ้า เมื่อ จึง ดังนั้น อย่างไรก็ตาม'))
        self.tokenizer = word_tokenize
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def basic_statistics(self) -> Dict[str, Any]:
        """คำนวณสถิติพื้นฐานของข้อมูล"""
        text_lengths = self.data[self.text_column].apply(len)
        word_counts = self.data[self.text_column].apply(lambda x: len(self.tokenizer(normalize(x))))
        
        stats = {
            'text_length': {
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'mean': text_lengths.mean(),
                'median': text_lengths.median(),
                'std': text_lengths.std()
            },
            'word_count': {
                'min': word_counts.min(),
                'max': word_counts.max(),
                'mean': word_counts.mean(),
                'median': word_counts.median(),
                'std': word_counts.std()
            }
        }
        
        return stats
    
    def plot_length_distribution(self, save_fig: bool = True) -> None:
        """สร้างกราฟแสดงการกระจายความยาวของข้อความ"""
        text_lengths = self.data[self.text_column].apply(len)
        
        plt.figure(figsize=(12, 8))
        sns.histplot(text_lengths, kde=True, bins=30)
        plt.title('การกระจายความยาวของข้อความ')
        plt.xlabel('ความยาวข้อความ (จำนวนตัวอักษร)')
        plt.ylabel('จำนวนตัวอย่าง')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_label_distribution(self, save_fig: bool = True) -> None:
        """สร้างกราฟแสดงการกระจายของ label"""
        if not self.label_column:
            return
        
        plt.figure(figsize=(12, 8))
        sns.countplot(x=self.label_column, data=self.data, palette="viridis")
        plt.title('การกระจายของ Label')
        plt.xlabel('Label')
        plt.ylabel('จำนวนตัวอย่าง')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_vocabulary(self, top_n: int = 30) -> Dict[str, Any]:
        """วิเคราะห์คลังคำศัพท์"""
        all_text = ' '.join(self.data[self.text_column])
        normalized_text = normalize(all_text)
        words = self.tokenizer(normalized_text)
        
        # กรอง stopwords และคำที่เป็นตัวเลขออก
        filtered_words = [word for word in words if word not in self.stopwords and not word.isdigit()]
        
        # นับความถี่ของคำ
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)
        
        vocab_data = {
            'vocabulary_size': len(word_counts),
            'top_words': [{'word': word, 'frequency': freq} for word, freq in top_words],
            'average_word_length': np.mean([len(word) for word in filtered_words])
        }
        
        return vocab_data

    def plot_word_frequency(self, top_n: int = 30, save_fig: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """สร้างกราฟแสดงความถี่ของคำที่พบบ่อย"""
        vocab_data = self.analyze_vocabulary(top_n=top_n)
        top_words = vocab_data["top_words"]
        
        # สร้างกราฟ
        fig, ax = plt.subplots(figsize=(12, 8))
        words = [item['word'] for item in top_words]
        freqs = [item['frequency'] for item in top_words]
        
        bars = sns.barplot(x=freqs, y=words, ax=ax, palette="viridis")
        ax.set_title(f'{top_n} คำที่พบบ่อยที่สุด')
        ax.set_xlabel('ความถี่')
        ax.set_ylabel('คำ')
        
        # เพิ่มตัวเลขความถี่ท้ายแท่ง
        for i, (freq, bar) in enumerate(zip(freqs, bars.patches)):
            bars.text(freq + 0.3, i, f"{freq}", ha='left', va='center')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'word_frequency.png'), dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def create_wordcloud(self, min_word_length: int = 2, save_fig: bool = True) -> plt.Figure:
        """สร้าง wordcloud จากข้อความ"""
        # รวมข้อความทั้งหมด
        all_text = ' '.join(self.data[self.text_column])
        
        # Normalize และแบ่งคำ
        normalized_text = normalize(all_text)
        words = self.tokenizer(normalized_text)
        
        # กรองคำที่สั้นเกินไปและ stopwords ออก
        filtered_words = [word for word in words 
                         if len(word) >= min_word_length 
                         and word not in self.stopwords
                         and not word.isdigit()
                         and not re.match(r'^[!@#$%^&*(),.;:\'"\/\[\]{}|<>]+$', word)]
        
        # สร้าง text สำหรับ wordcloud
        text = ' '.join(filtered_words)
        
        # สร้าง wordcloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=200,
                             colormap='viridis',
                             collocations=False,
                             font_path='font/THSarabunNew.ttf' if os.path.exists('font/THSarabunNew.ttf') else None).generate(text)
        
        # สร้างกราฟ
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'wordcloud.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_pos_distribution(self, save_fig: bool = True) -> Dict[str, Any]:
        """วิเคราะห์การกระจายของชนิดคำ (Part of Speech)"""
        all_pos_tags = []
        
        # สร้าง progress bar
        for text in tqdm(self.data[self.text_column], desc="POS Tagging"):
            # Normalize ข้อความก่อน tokenize
            normalized_text = normalize(text)
            # แบ่งคำและหา POS
            words = self.tokenizer(normalized_text)
            tags = pos_tag(words, engine='perceptron')
            pos_tags = [tag for _, tag in tags]
            all_pos_tags.extend(pos_tags)
        
        # นับความถี่ของ POS
        pos_counts = Counter(all_pos_tags)
        top_pos = pos_counts.most_common()
        
        # สร้างแผนภูมิ
        fig, ax = plt.subplots(figsize=(12, 8))
        labels = [tag for tag, _ in top_pos]
        counts = [count for _, count in top_pos]
        
        bars = sns.barplot(x=labels, y=counts, ax=ax, palette="rocket")
        ax.set_title('การกระจายของชนิดคำ (Part of Speech)')
        ax.set_xlabel('ชนิดคำ')
        ax.set_ylabel('จำนวน')
        plt.xticks(rotation=45, ha='right')
        
        # เพิ่มตัวเลขความถี่บนแท่ง
        for bar in bars.patches:
            bars.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + 0.3, 
                    f"{int(bar.get_height())}", 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'pos_distribution.png'), dpi=300, bbox_inches='tight')
        
        # จัดเตรียมข้อมูลสำหรับ return
        pos_data = {
            'total_tokens': len(all_pos_tags),
            'pos_counts': {tag: count for tag, count in top_pos},
            'plot_path': os.path.join(self.output_dir, 'pos_distribution.png') if save_fig else None
        }
        
        return pos_data
    
    def analyze_text_similarity(self, n_clusters: int = 5, method: str = 'tfidf', save_fig: bool = True) -> Dict[str, Any]:
        """วิเคราะห์ความคล้ายคลึงระหว่างข้อความ"""
        # เตรียมข้อความทั้งหมด
        texts = self.data[self.text_column].tolist()
        
        if len(texts) < 2:
            return {"error": "ต้องมีข้อความอย่างน้อย 2 ข้อความขึ้นไปสำหรับการวิเคราะห์ความคล้ายคลึง"}
        
        # วิธีการสร้าง feature vector
        if method == 'tfidf':
            # ใช้ TF-IDF
            vectorizer = TfidfVectorizer(
                tokenizer=self.tokenizer,
                stop_words=self.stopwords,
                min_df=2,
                max_df=0.95
            )
            X = vectorizer.fit_transform(texts)
            
        elif method == 'count':
            # ใช้ Count Vectorizer
            vectorizer = CountVectorizer(
                tokenizer=self.tokenizer,
                stop_words=self.stopwords,
                min_df=2,
                max_df=0.95
            )
            X = vectorizer.fit_transform(texts)
            
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'tfidf' or 'count'.")
        
        # ลดมิติข้อมูลด้วย TruncatedSVD
        n_components = min(100, X.shape[1] - 1, X.shape[0] - 1)
        if n_components <= 0:
            n_components = 1
            
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X)
        
        # หาจำนวน clusters ที่เหมาะสมโดยใช้ silhouette score
        if len(texts) > 3 and n_clusters > 1:  # ต้องมีข้อความมากกว่า 3 ข้อความ
            range_n_clusters = range(2, min(n_clusters + 1, len(texts)))
            silhouette_scores = []
            
            for n in range_n_clusters:
                clusterer = KMeans(n_clusters=n, random_state=42)
                cluster_labels = clusterer.fit_predict(X_reduced)
                
                try:
                    silhouette_avg = silhouette_score(X_reduced, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                except:
                    silhouette_scores.append(-1)
            
            # เลือกจำนวน cluster ที่ให้ค่า silhouette สูงสุด
            best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)] if silhouette_scores else n_clusters
        else:
            best_n_clusters = 1 if len(texts) <= 3 else n_clusters
        
        # ทำ clustering
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_reduced)
        
        # เพิ่ม cluster labels เข้าไปในข้อมูล
        cluster_df = self.data.copy()
        cluster_df['cluster'] = clusters
        
        # คำนวณ similarity matrix
        similarity_matrix = cosine_similarity(X)
        
        # ลดมิติข้อมูลเพื่อแสดงผลด้วย t-SNE
        if len(texts) > 2:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
                X_tsne = tsne.fit_transform(X_reduced)
                
                # สร้างกราฟ t-SNE
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Cluster')
                plt.title('การกระจายของข้อความโดยใช้ t-SNE')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                
                # เพิ่มหมายเลขข้อความ
                for i, (x, y) in enumerate(X_tsne):
                    plt.annotate(str(i), (x, y), fontsize=8, alpha=0.7)
                
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(os.path.join(self.output_dir, 'text_tsne.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                tsne_path = os.path.join(self.output_dir, 'text_tsne.png') if save_fig else None
            except Exception as e:
                print(f"Error creating t-SNE plot: {str(e)}")
                tsne_path = None
        else:
            tsne_path = None
        
        # สร้าง heatmap สำหรับ similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='YlGnBu', xticklabels=False, yticklabels=False)
        plt.title('Similarity Matrix')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'similarity_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # หาข้อความที่คล้ายกันมากที่สุด
        np.fill_diagonal(similarity_matrix, 0)  # ตัด self-similarity ออก
        most_similar_pairs = []
        
        for i in range(len(texts)):
            similarities = similarity_matrix[i]
            if len(similarities) > 0:
                most_similar_idx = np.argmax(similarities)
                similarity_score = similarities[most_similar_idx]
                
                if similarity_score > 0.3:  # กำหนดค่า threshold
                    most_similar_pairs.append({
                        'text1_idx': i,
                        'text2_idx': most_similar_idx,
                        'similarity': float(similarity_score),
                        'text1': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                        'text2': texts[most_similar_idx][:100] + '...' if len(texts[most_similar_idx]) > 100 else texts[most_similar_idx]
                    })
        
        # เรียงลำดับตามความคล้ายคลึง
        most_similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # จำกัดจำนวนคู่
        most_similar_pairs = most_similar_pairs[:20]
        
        # สรุปผลลัพธ์
        summary = {
            'best_n_clusters': best_n_clusters,
            'cluster_sizes': {i: int((clusters == i).sum()) for i in range(best_n_clusters)},
            'silhouette_scores': {n: score for n, score in zip(range(2, min(n_clusters + 1, len(texts))), silhouette_scores)} if len(texts) > 3 else {},
            'most_similar_pairs': most_similar_pairs,
            'plots': {
                'tsne': tsne_path,
                'similarity_matrix': os.path.join(self.output_dir, 'similarity_matrix.png') if save_fig else None
            }
        }
        
        return summary
    
    def analyze_content_complexity(self, save_fig: bool = True) -> Dict[str, Any]:
        """วิเคราะห์ความซับซ้อนของเนื้อหา"""
        # คำนวณความหลากหลายของคำศัพท์ (Lexical Diversity)
        lexical_diversity = []
        sentence_lengths = []
        word_lengths = []
        
        for text in self.data[self.text_column]:
            normalized_text = normalize(text)
            words = self.tokenizer(normalized_text)
            
            # ความหลากหลายของคำศัพท์: จำนวนคำไม่ซ้ำ / จำนวนคำทั้งหมด
            if len(words) > 0:
                diversity = len(set(words)) / len(words)
                lexical_diversity.append(diversity)
            
            # ความยาวเฉลี่ยของประโยค
            sentences = text.split('.')
            valid_sentences = [s.strip() for s in sentences if s.strip()]
            if valid_sentences:
                avg_sent_length = np.mean([len(self.tokenizer(normalize(s))) for s in valid_sentences])
                sentence_lengths.append(avg_sent_length)
            
            # ความยาวเฉลี่ยของคำ
            if words:
                avg_word_length = np.mean([len(w) for w in words])
                word_lengths.append(avg_word_length)
        
        # คำนวณความซับซ้อนโครงสร้างประโยค
        complexity_scores = []
        for text in self.data[self.text_column]:
            # นับจำนวนคำเชื่อม
            connectives = ['และ', 'หรือ', 'แต่', 'เพราะ', 'เพื่อ', 'ถ้า', 'เมื่อ', 'จึง', 'ดังนั้น', 'อย่างไรก็ตาม']
            connective_count = sum(1 for c in connectives if c in text)
            
            # นับจำนวนเครื่องหมายวรรคตอน
            punctuation_count = sum(1 for c in text if c in ',;():')
            
            # คำนวณคะแนนความซับซ้อน
            words = self.tokenizer(normalize(text))
            if len(words) > 0:
                complexity = (connective_count + punctuation_count) / len(words)
                complexity_scores.append(complexity)
        
        # สร้างกราฟแสดงความหลากหลายของคำศัพท์
        plt.figure(figsize=(12, 8))
        sns.histplot(lexical_diversity, kde=True, bins=30)
        plt.title('การกระจายความหลากหลายของคำศัพท์')
        plt.xlabel('Lexical Diversity (จำนวนคำไม่ซ้ำ / จำนวนคำทั้งหมด)')
        plt.ylabel('จำนวนตัวอย่าง')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'lexical_diversity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # สร้างกราฟแสดงความสัมพันธ์ระหว่างความยาวประโยคและความซับซ้อน
        plt.figure(figsize=(10, 6))
        plt.scatter(sentence_lengths, complexity_scores, alpha=0.6)
        plt.title('ความสัมพันธ์ระหว่างความยาวประโยคและความซับซ้อน')
        plt.xlabel('ความยาวประโยคเฉลี่ย (จำนวนคำต่อประโยค)')
        plt.ylabel('คะแนนความซับซ้อน')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'sentence_complexity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # สรุปข้อมูล
        result = {
            'lexical_diversity': {
                'mean': np.mean(lexical_diversity) if lexical_diversity else None,
                'median': np.median(lexical_diversity) if lexical_diversity else None,
                'std': np.std(lexical_diversity) if lexical_diversity else None,
                'plot': os.path.join(self.output_dir, 'lexical_diversity.png') if save_fig else None
            },
            'sentence_length': {
                'mean': np.mean(sentence_lengths) if sentence_lengths else None,
                'median': np.median(sentence_lengths) if sentence_lengths else None,
                'std': np.std(sentence_lengths) if sentence_lengths else None
            },
            'word_length': {
                'mean': np.mean(word_lengths) if word_lengths else None,
                'median': np.median(word_lengths) if word_lengths else None,
                'std': np.std(word_lengths) if word_lengths else None
            },
            'complexity': {
                'mean': np.mean(complexity_scores) if complexity_scores else None,
                'median': np.median(complexity_scores) if complexity_scores else None,
                'std': np.std(complexity_scores) if complexity_scores else None,
                'plot': os.path.join(self.output_dir, 'sentence_complexity.png') if save_fig else None
            }
        }
        
        return result
    
    def analyze_label_correlation(self, save_fig: bool = True) -> Optional[Dict[str, Any]]:
        """วิเคราะห์ความสัมพันธ์ระหว่าง label และลักษณะของข้อความ"""
        if not self.label_column:
            return None
        
        result = {}
        
        # แปลง label เป็นประเภท categorical
        df = self.data.copy()
        df[self.label_column] = df[self.label_column].astype('category')
        
        # ความสัมพันธ์ระหว่าง label และความยาวข้อความ
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x=self.label_column, y='text_length', data=df)
        plt.title('ความสัมพันธ์ระหว่าง Label และความยาวข้อความ')
        plt.xlabel(self.label_column)
        plt.ylabel('ความยาวข้อความ (จำนวนตัวอักษร)')
        plt.xticks(rotation=45)
        
        # เพิ่มข้อมูลสถิติ
        label_means = df.groupby(self.label_column)['text_length'].mean()
        for i, label in enumerate(label_means.index):
            ax.text(i, label_means[label] + 5, f'{label_means[label]:.1f}', 
                   ha='center', va='bottom', alpha=0.7)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'label_text_length.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        result['length_by_label'] = {
            'mean': df.groupby(self.label_column)['text_length'].mean().to_dict(),
            'median': df.groupby(self.label_column)['text_length'].median().to_dict(),
            'std': df.groupby(self.label_column)['text_length'].std().to_dict(),
            'plot': os.path.join(self.output_dir, 'label_text_length.png') if save_fig else None
        }
        
        # ความสัมพันธ์ระหว่าง label และจำนวนคำ
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x=self.label_column, y='word_count', data=df)
        plt.title('ความสัมพันธ์ระหว่าง Label และจำนวนคำ')
        plt.xlabel(self.label_column)
        plt.ylabel('จำนวนคำ')
        plt.xticks(rotation=45)
        
        # เพิ่มข้อมูลสถิติ
        label_means = df.groupby(self.label_column)['word_count'].mean()
        for i, label in enumerate(label_means.index):
            ax.text(i, label_means[label] + 1, f'{label_means[label]:.1f}', 
                   ha='center', va='bottom', alpha=0.7)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'label_word_count.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        result['words_by_label'] = {
            'mean': df.groupby(self.label_column)['word_count'].mean().to_dict(),
            'median': df.groupby(self.label_column)['word_count'].median().to_dict(),
            'std': df.groupby(self.label_column)['word_count'].std().to_dict(),
            'plot': os.path.join(self.output_dir, 'label_word_count.png') if save_fig else None
        }
        
        # วิเคราะห์คำสำคัญที่พบมากในแต่ละ label
        label_keywords = {}
        
        for label in df[self.label_column].unique():
            # ดึงข้อความที่มี label นี้
            texts = df[df[self.label_column] == label][self.text_column].tolist()
            
            # สร้าง corpus
            corpus = []
            for text in texts:
                normalized_text = normalize(text)
                words = [w for w in self.tokenizer(normalized_text) 
                         if w not in self.stopwords 
                         and len(w) > 1 
                         and not w.isdigit()]
                corpus.append(' '.join(words))
            
            if not corpus:
                continue
            
            # ใช้ TF-IDF เพื่อหาคำสำคัญ
            try:
                vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
                X = vectorizer.fit_transform(corpus)
                
                # ดึงคำและ TF-IDF scores
                feature_names = vectorizer.get_feature_names_out()
                
                if len(feature_names) == 0:
                    continue
                
                # หาค่า TF-IDF เฉลี่ยของแต่ละคำ
                tfidf_means = X.mean(axis=0).A1
                
                # จัดลำดับตามค่า TF-IDF
                sorted_indices = np.argsort(tfidf_means)[::-1]
                
                # เลือก 10 คำแรก
                top_indices = sorted_indices[:10]
                top_words = [feature_names[i] for i in top_indices]
                top_scores = [float(tfidf_means[i]) for i in top_indices]
                
                label_keywords[str(label)] = [
                    {'word': word, 'score': score}
                    for word, score in zip(top_words, top_scores)
                ]
            except Exception as e:
                print(f"Error analyzing keywords for label '{label}': {str(e)}")
        
        result['keywords_by_label'] = label_keywords
        
        return result
    
    def generate_report(self, output_format: str = 'json') -> str:
        """สร้างรายงานการวิเคราะห์ข้อมูล"""
        # เก็บรวบรวมข้อมูลทั้งหมด
        report_data = {
            'dataset_info': {
                'filename': getattr(self.data, 'name', 'Unknown'),
                'samples': len(self.data),
                'columns': list(self.data.columns),
                'text_column': self.text_column,
                'label_column': self.label_column
            },
            'basic_statistics': self.basic_statistics(),
            'vocabulary_analysis': self.analyze_vocabulary(),
            'pos_distribution': self.analyze_pos_distribution(save_fig=True),
            'content_complexity': self.analyze_content_complexity(save_fig=True)
        }
        
        # เพิ่มข้อมูลการวิเคราะห์ label ถ้ามี
        if self.label_column:
            report_data['label_analysis'] = {
                'distribution': self.data[self.label_column].value_counts().to_dict(),
                'correlation': self.analyze_label_correlation(save_fig=True)
            }
        
        # เพิ่มข้อมูล text similarity ถ้ามีข้อความมากกว่า 1
        if len(self.data) > 1:
            report_data['text_similarity'] = self.analyze_text_similarity(save_fig=True)
        
        # บันทึกข้อมูลรายงาน
        if output_format == 'json':
            output_path = os.path.join(self.output_dir, 'analysis_report.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            return output_path
        
        elif output_format == 'html':
            output_path = os.path.join(self.output_dir, 'analysis_report.html')
            
            # สร้าง HTML report
            html_content = [
                '<!DOCTYPE html>',
                '<html lang="th">',
                '<head>',
                '<meta charset="UTF-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                '<title>รายงานการวิเคราะห์ข้อมูล</title>',
                '<style>',
                'body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }',
                'h1, h2, h3 { color: #333366; }',
                'table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
                'th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }',
                'th { background-color: #f2f2f2; }',
                'tr:nth-child(even) { background-color: #f9f9f9; }',
                'img { max-width: 100%; height: auto; margin: 10px 0; }',
                '.section { margin-bottom: 30px; border: 1px solid #eee; padding: 20px; border-radius: 5px; }',
                '</style>',
                '</head>',
                '<body>',
                '<h1>รายงานการวิเคราะห์ข้อมูล</h1>'
            ]
            
            # ข้อมูลทั่วไป
            html_content.extend([
                '<div class="section">',
                '<h2>ข้อมูลชุดข้อมูล</h2>',
                f'<p>จำนวนตัวอย่าง: {report_data["dataset_info"]["samples"]}</p>',
                f'<p>คอลัมน์ข้อความ: {report_data["dataset_info"]["text_column"]}</p>'
            ])
            
            if report_data["dataset_info"]["label_column"]:
                html_content.append(f'<p>คอลัมน์ label: {report_data["dataset_info"]["label_column"]}</p>')
            
            html_content.append('</div>')
            
            # สถิติพื้นฐาน
            html_content.extend([
                '<div class="section">',
                '<h2>สถิติพื้นฐาน</h2>',
                '<h3>ความยาวข้อความ</h3>',
                '<table>',
                '<tr><th>สถิติ</th><th>ค่า</th></tr>',
                f'<tr><td>ค่าน้อยสุด</td><td>{report_data["basic_statistics"]["text_length"]["min"]}</td></tr>',
                f'<tr><td>ค่ามากสุด</td><td>{report_data["basic_statistics"]["text_length"]["max"]}</td></tr>',
                f'<tr><td>ค่าเฉลี่ย</td><td>{report_data["basic_statistics"]["text_length"]["mean"]:.2f}</td></tr>',
                f'<tr><td>ค่ามัธยฐาน</td><td>{report_data["basic_statistics"]["text_length"]["median"]:.2f}</td></tr>',
                f'<tr><td>ส่วนเบี่ยงเบนมาตรฐาน</td><td>{report_data["basic_statistics"]["text_length"]["std"]:.2f}</td></tr>',
                '</table>',
                
                '<h3>จำนวนคำ</h3>',
                '<table>',
                '<tr><th>สถิติ</th><th>ค่า</th></tr>',
                f'<tr><td>ค่าน้อยสุด</td><td>{report_data["basic_statistics"]["word_count"]["min"]}</td></tr>',
                f'<tr><td>ค่ามากสุด</td><td>{report_data["basic_statistics"]["word_count"]["max"]}</td></tr>',
                f'<tr><td>ค่าเฉลี่ย</td><td>{report_data["basic_statistics"]["word_count"]["mean"]:.2f}</td></tr>',
                f'<tr><td>ค่ามัธยฐาน</td><td>{report_data["basic_statistics"]["word_count"]["median"]:.2f}</td></tr>',
                f'<tr><td>ส่วนเบี่ยงเบนมาตรฐาน</td><td>{report_data["basic_statistics"]["word_count"]["std"]:.2f}</td></tr>',
                '</table>',
                
                '<img src="length_distribution.png" alt="การกระจายความยาวข้อความ">',
                '</div>'
            ])
            
            # การวิเคราะห์คำศัพท์
            html_content.extend([
                '<div class="section">',
                '<h2>การวิเคราะห์คำศัพท์</h2>',
                f'<p>ขนาดคลังคำศัพท์: {report_data["vocabulary_analysis"]["vocabulary_size"]} คำ</p>',
                f'<p>ความยาวคำเฉลี่ย: {report_data["vocabulary_analysis"]["average_word_length"]:.2f} ตัวอักษร</p>',
                
                '<h3>คำที่พบบ่อย</h3>',
                '<table>',
                '<tr><th>คำ</th><th>ความถี่</th></tr>'
            ])
            
            for word_data in report_data["vocabulary_analysis"]["top_words"][:20]:
                html_content.append(f'<tr><td>{word_data["word"]}</td><td>{word_data["frequency"]}</td></tr>')
            
            html_content.extend([
                '</table>',
                '<img src="word_frequency.png" alt="ความถี่ของคำที่พบบ่อย">',
                '<img src="wordcloud.png" alt="Word Cloud">',
                '</div>'
            ])
            
            # การกระจายของ Part of Speech
            html_content.extend([
                '<div class="section">',
                '<h2>การกระจายของชนิดคำ (Part of Speech)</h2>',
                f'<p>จำนวน token ทั้งหมด: {report_data["pos_distribution"]["total_tokens"]}</p>',
                '<table>',
                '<tr><th>ชนิดคำ</th><th>จำนวน</th></tr>'
            ])
            
            for pos, count in list(report_data["pos_distribution"]["pos_counts"].items())[:15]:
                html_content.append(f'<tr><td>{pos}</td><td>{count}</td></tr>')
            
            html_content.extend([
                '</table>',
                '<img src="pos_distribution.png" alt="การกระจายของชนิดคำ">',
                '</div>'
            ])
            
            # ความซับซ้อนของเนื้อหา
            html_content.extend([
                '<div class="section">',
                '<h2>ความซับซ้อนของเนื้อหา</h2>',
                
                '<h3>ความหลากหลายของคำศัพท์</h3>',
                '<table>',
                '<tr><th>สถิติ</th><th>ค่า</th></tr>',
                f'<tr><td>ค่าเฉลี่ย</td><td>{report_data["content_complexity"]["lexical_diversity"]["mean"]:.4f}</td></tr>',
                f'<tr><td>ค่ามัธยฐาน</td><td>{report_data["content_complexity"]["lexical_diversity"]["median"]:.4f}</td></tr>',
                f'<tr><td>ส่วนเบี่ยงเบนมาตรฐาน</td><td>{report_data["content_complexity"]["lexical_diversity"]["std"]:.4f}</td></tr>',
                '</table>',
                
                '<h3>ความยาวประโยค</h3>',
                '<table>',
                '<tr><th>สถิติ</th><th>ค่า</th></tr>',
                f'<tr><td>ค่าเฉลี่ย</td><td>{report_data["content_complexity"]["sentence_length"]["mean"]:.2f} คำ/ประโยค</td></tr>',
                f'<tr><td>ค่ามัธยฐาน</td><td>{report_data["content_complexity"]["sentence_length"]["median"]:.2f} คำ/ประโยค</td></tr>',
                f'<tr><td>ส่วนเบี่ยงเบนมาตรฐาน</td><td>{report_data["content_complexity"]["sentence_length"]["std"]:.2f} คำ/ประโยค</td></tr>',
                '</table>',
                
                '<img src="lexical_diversity.png" alt="การกระจายความหลากหลายของคำศัพท์">',
                '<img src="sentence_complexity.png" alt="ความสัมพันธ์ระหว่างความยาวประโยคและความซับซ้อน">',
                '</div>'
            ])
            
            # การวิเคราะห์ label
            if 'label_analysis' in report_data:
                html_content.extend([
                    '<div class="section">',
                    '<h2>การวิเคราะห์ Label</h2>',
                    
                    '<h3>การกระจายของ Label</h3>',
                    '<table>',
                    '<tr><th>Label</th><th>จำนวน</th></tr>'
                ])
                
                for label, count in report_data["label_analysis"]["distribution"].items():
                    html_content.append(f'<tr><td>{label}</td><td>{count}</td></tr>')
                
                html_content.append('</table>')
                
                if report_data["label_analysis"]["correlation"]:
                    html_content.extend([
                        '<h3>ความสัมพันธ์ระหว่าง Label และความยาวข้อความ</h3>',
                        '<img src="label_text_length.png" alt="ความสัมพันธ์ระหว่าง Label และความยาวข้อความ">',
                        
                        '<h3>ความสัมพันธ์ระหว่าง Label และจำนวนคำ</h3>',
                        '<img src="label_word_count.png" alt="ความสัมพันธ์ระหว่าง Label และจำนวนคำ">',
                        
                        '<h3>คำสำคัญในแต่ละ Label</h3>'
                    ])
                    
                    for label, keywords in report_data["label_analysis"]["correlation"]["keywords_by_label"].items():
                        html_content.extend([
                            f'<h4>Label: {label}</h4>',
                            '<table>',
                            '<tr><th>คำสำคัญ</th><th>คะแนน</th></tr>'
                        ])
                        
                        for keyword in keywords:
                            html_content.append(f'<tr><td>{keyword["word"]}</td><td>{keyword["score"]:.4f}</td></tr>')
                        
                        html_content.append('</table>')
                
                html_content.append('</div>')
            
            # ความคล้ายคลึงระหว่างข้อความ
            if 'text_similarity' in report_data:
                html_content.extend([
                    '<div class="section">',
                    '<h2>ความคล้ายคลึงระหว่างข้อความ</h2>',
                    f'<p>จำนวน cluster ที่เหมาะสม: {report_data["text_similarity"]["best_n_clusters"]}</p>',
                    
                    '<h3>ขนาดของแต่ละ Cluster</h3>',
                    '<table>',
                    '<tr><th>Cluster</th><th>จำนวนตัวอย่าง</th></tr>'
                ])
                
                for cluster, size in report_data["text_similarity"]["cluster_sizes"].items():
                    html_content.append(f'<tr><td>{cluster}</td><td>{size}</td></tr>')
                
                html_content.append('</table>')
                
                # รูปภาพ t-SNE และ similarity matrix
                if report_data["text_similarity"]["plots"]["tsne"]:
                    html_content.append('<img src="text_tsne.png" alt="การกระจายของข้อความด้วย t-SNE">')
                
                html_content.append('<img src="similarity_matrix.png" alt="Similarity Matrix">')
                
                # คู่ข้อความที่คล้ายกันมากที่สุด
                if report_data["text_similarity"]["most_similar_pairs"]:
                    html_content.extend([
                        '<h3>คู่ข้อความที่คล้ายกันมากที่สุด</h3>',
                        '<table>',
                        '<tr><th>ข้อความที่ 1</th><th>ข้อความที่ 2</th><th>คะแนนความคล้ายคลึง</th></tr>'
                    ])
                    
                    for pair in report_data["text_similarity"]["most_similar_pairs"][:10]:
                        html_content.append(
                            f'<tr><td>{pair["text1"]}</td><td>{pair["text2"]}</td><td>{pair["similarity"]:.4f}</td></tr>'
                        )
                    
                    html_content.append('</table>')
                
                html_content.append('</div>')
            
            # ปิด HTML
            html_content.extend([
                '</body>',
                '</html>'
            ])
            
            # บันทึก HTML
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            
            return output_path
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}. Use 'json' or 'html'.")

def main():
    parser = argparse.ArgumentParser(description='วิเคราะห์และสรุปข้อมูลภาษาไทย')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='explainer_output', help='Output directory')
    parser.add_argument('--text_column', type=str, default='text', help='Column containing text data')
    parser.add_argument('--label_column', type=str, default=None, help='Column containing labels (if any)')
    parser.add_argument('--output_format', type=str, choices=['json', 'html'], default='html', help='Output format')
    
    args = parser.parse_args()
    
    # สร้าง DataExplainer และวิเคราะห์ข้อมูล
    explainer = DataExplainer(
        data=args.input_file,
        text_column=args.text_column,
        label_column=args.label_column,
        output_dir=args.output_dir
    )
    
    # สร้างกราฟและวิเคราะห์ข้อมูล
    print("Analyzing data...")
    
    print("1. Analyzing basic statistics...")
    explainer.basic_statistics()
    
    print("2. Plotting length distribution...")
    explainer.plot_length_distribution()
    
    print("3. Plotting label distribution...")
    if args.label_column:
        explainer.plot_label_distribution()
    
    print("4. Analyzing vocabulary...")
    explainer.analyze_vocabulary()
    
    print("5. Creating wordcloud...")
    explainer.create_wordcloud()
    
    print("6. Plotting word frequency...")
    explainer.plot_word_frequency()
    
    print("7. Analyzing part-of-speech distribution...")
    explainer.analyze_pos_distribution()
    
    print("8. Analyzing content complexity...")
    explainer.analyze_content_complexity()
    
    print("9. Analyzing text similarity...")
    if len(explainer.data) > 1:
        explainer.analyze_text_similarity()
    
    print("10. Analyzing label correlation...")
    if args.label_column:
        explainer.analyze_label_correlation()
    
    # สร้างรายงาน
    print("Generating report...")
    report_path = explainer.generate_report(output_format=args.output_format)
    
    print(f"\nAnalysis complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()