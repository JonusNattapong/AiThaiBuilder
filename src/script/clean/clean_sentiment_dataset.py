import pandas as pd
import argparse
from collections import Counter
import os

def remove_duplicates_maintain_balance(df, target_per_class=None):
    """Remove duplicates while maintaining class balance"""
    # Keep first occurrence of duplicates
    unique_df = df.drop_duplicates(subset=['text'], keep='first')
    
    # Count samples per class
    class_counts = unique_df['label'].value_counts()
    min_count = min(class_counts) if target_per_class is None else target_per_class
    
    # Balance classes
    balanced_dfs = []
    for label in class_counts.index:
        class_df = unique_df[unique_df['label'] == label]
        balanced_dfs.append(class_df.head(min_count))
    
    # Combine balanced data
    cleaned_df = pd.concat(balanced_dfs)
    cleaned_df = cleaned_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return cleaned_df

def main():
    parser = argparse.ArgumentParser(description='Clean sentiment dataset')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output cleaned CSV file')
    parser.add_argument('--samples_per_class', type=int, default=None,
                      help='Target number of samples per class')
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.input_file)
    print(f"\nOriginal dataset size: {len(df)}")
    print("Original class distribution:")
    print(df['label'].value_counts())
    print(f"\nDuplicate texts: {len(df[df.duplicated(['text'], keep=False)])}")
    
    # Clean dataset
    cleaned_df = remove_duplicates_maintain_balance(df, args.samples_per_class)
    
    # Save cleaned dataset
    cleaned_df.to_csv(args.output_file, index=False)
    
    print(f"\nCleaned dataset size: {len(cleaned_df)}")
    print("Cleaned class distribution:")
    print(cleaned_df['label'].value_counts())
    print(f"\nCleaned dataset saved to: {args.output_file}")

if __name__ == "__main__":
    main()