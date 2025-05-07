import pandas as pd
import numpy as np
from collections import Counter
import re
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

def load_dataset(file_path):
    """Load and validate dataset structure"""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['text', 'label']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def check_class_distribution(df):
    """Check distribution of labels"""
    print("\n=== Label Distribution ===")
    dist = df['label'].value_counts()
    percentages = (dist / len(df) * 100).round(2)
    
    for label, count in dist.items():
        print(f"{label}: {count} samples ({percentages[label]}%)")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.close()

def check_text_statistics(df):
    """Analyze text length statistics"""
    print("\n=== Text Statistics ===")
    
    # Calculate lengths
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    
    # Print statistics
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    print(f"Median text length: {df['text_length'].median():.1f} characters")
    print(f"Average word count: {df['word_count'].mean():.1f} words")
    print(f"Median word count: {df['word_count'].median():.1f} words")
    
    # Plot length distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(data=df, x='text_length', bins=50, ax=ax1)
    ax1.set_title('Text Length Distribution')
    
    sns.histplot(data=df, x='word_count', bins=50, ax=ax2)
    ax2.set_title('Word Count Distribution')
    
    plt.tight_layout()
    plt.savefig('text_statistics.png')
    plt.close()

def check_thai_text_validity(df):
    """Check if text contains valid Thai characters"""
    print("\n=== Thai Text Validation ===")
    
    thai_pattern = re.compile(r'[ก-๙]')
    
    def has_thai(text):
        return bool(thai_pattern.search(text))
    
    valid_thai = df['text'].apply(has_thai)
    invalid_samples = df[~valid_thai]
    
    print(f"Total samples with Thai text: {valid_thai.sum()} ({valid_thai.mean()*100:.1f}%)")
    print(f"Samples without Thai text: {len(invalid_samples)} ({(~valid_thai).mean()*100:.1f}%)")
    
    if len(invalid_samples) > 0:
        print("\nSample rows without Thai text:")
        for idx, row in invalid_samples.head().iterrows():
            print(f"Index {idx}: {row['text'][:100]}...")

def check_duplicates(df):
    """Check for duplicate texts"""
    print("\n=== Duplicate Analysis ===")
    
    # Exact duplicates
    exact_dupes = df[df.duplicated(['text'], keep=False)]
    print(f"Exact duplicate texts: {len(exact_dupes)} ({len(exact_dupes)/len(df)*100:.1f}%)")
    
    # Near duplicates (after normalization)
    df['normalized_text'] = df['text'].apply(normalize)
    norm_dupes = df[df.duplicated(['normalized_text'], keep=False)]
    print(f"Near-duplicate texts (after normalization): {len(norm_dupes)} ({len(norm_dupes)/len(df)*100:.1f}%)")
    
    if len(exact_dupes) > 0:
        print("\nSample duplicate texts:")
        for text, group in exact_dupes.groupby('text').head(3).groupby('text'):
            print(f"\nText: {text[:100]}...")
            print(f"Appears {len(group)} times with labels: {group['label'].tolist()}")

def check_null_values(df):
    """Check for null or empty values"""
    print("\n=== Missing Value Analysis ===")
    
    nulls = df.isnull().sum()
    print("\nNull values per column:")
    for col, count in nulls.items():
        print(f"{col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check for empty strings
    empties = df.astype(str).apply(lambda x: x.str.strip().eq('').sum())
    print("\nEmpty strings per column:")
    for col, count in empties.items():
        print(f"{col}: {count} ({count/len(df)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Check dataset quality')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    args = parser.parse_args()
    
    # Load dataset
    df = load_dataset(args.input_file)
    if df is None:
        return
    
    print(f"\nDataset loaded: {len(df)} samples")
    
    # Run quality checks
    check_class_distribution(df)
    check_text_statistics(df)
    check_thai_text_validity(df)
    check_duplicates(df)
    check_null_values(df)
    
    print("\nQuality check completed! See generated plots for visualizations.")

if __name__ == "__main__":
    main()