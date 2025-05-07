import argparse
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """Compute accuracy metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_sentiment_model(args):
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Create label mapping
    unique_labels = df['label'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    labels = df['label'].map(label_map).values
    texts = df['text'].tolist()
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, shuffle=True
    )

    # Load tokenizer and model with auth token if provided
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token if args.hf_token else None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(unique_labels),
        token=args.hf_token if args.hf_token else None
    )

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        do_eval=True,
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model and tokenizer
    final_model_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save label mapping if not exists
    label_map_save_path = os.path.join(args.output_dir, "label_map.json")
    if not os.path.exists(label_map_save_path):
        with open(label_map_save_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"\nTraining completed. Model saved to: {final_model_dir}")
    print(f"Label mapping saved to: {label_map_save_path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune model for sentiment analysis')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to sentiment dataset CSV')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                      help='Pretrained model to fine-tune')
    parser.add_argument('--hf_token', type=str, default=None,
                      help='Hugging Face token for accessing models')
    parser.add_argument('--output_dir', type=str, default='models/sentiment',
                      help='Output directory for model and checkpoints')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory relative to current working directory
    output_path = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Update args with absolute path
    args.output_dir = output_path
    
    train_sentiment_model(args)

if __name__ == "__main__":
    main()