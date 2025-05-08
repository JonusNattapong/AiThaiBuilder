import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import os

def clean_text(text):
    """Clean and normalize Thai text"""
    if not isinstance(text, str):
        return ""
    # Remove special characters but keep Thai characters and spaces
    text = re.sub(r'[^ก-๙\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

def validate_sentiment(row):
    """Validate sentiment labels and confidence scores"""
    valid_labels = ['angry', 'happy', 'sad']  # Update with actual labels from your dataset
    label = str(row['label']).lower()
    
    if label not in valid_labels:
        return False
    if not row['text'] or len(clean_text(row['text'])) < 10:  # Minimum text length
        return False
    return True

def prepare_dataset(input_file, output_dir):
    """Prepare sentiment dataset for fine-tuning"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the dataset
    print(f"Reading dataset from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Initial dataset size: {len(df)}")
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Validate and filter data
    valid_mask = df.apply(validate_sentiment, axis=1)
    df = df[valid_mask]
    print(f"Dataset size after cleaning: {len(df)}")
    
    # Convert labels to lowercase
    df['label'] = df['label'].str.lower()
    
    # Encode labels
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    
    # Create and save label mapping
    labels_list = sorted(df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(labels_list)}
    print("Label mapping:", label_mapping)
    
    # Convert labels using mapping
    df['label_id'] = df['label'].map(label_mapping)
    
    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'label_mapping': label_mapping,
        'class_distribution': df['label'].value_counts().to_dict(),
        'avg_text_length': float(df['text'].str.len().mean()),
        'num_classes': len(label_mapping)
    }
    
    print("\nDataset statistics:")
    print(f"Average text length: {metadata['avg_text_length']:.1f} characters")
    print("\nClass distribution:")
    for label_name, count in metadata['class_distribution'].items():
        print(f"{label_name}: {count} ({count/len(df)*100:.1f}%)")
    
    return metadata

if __name__ == "__main__":
    input_file = "DataOutput/thai_sentiment_deepseek_parallel.csv"
    output_dir = "DataOutput/processed_sentiment"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
    else:
        metadata = prepare_dataset(input_file, output_dir)
        print("\nDataset preparation completed.")