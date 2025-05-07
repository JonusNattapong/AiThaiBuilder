import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import argparse

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def save_model_safetensors(model, output_dir):
    """Save model weights in safetensors format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Convert to safetensors format and save
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

def train_model(args):
    # Load data
    df = pd.read_csv(args.input_file)
    
    # Convert labels to numeric
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label_id'] = df['label'].map(label_map)
    
    # Save label map
    label_map_file = os.path.join(args.output_dir, "label_map.json")
    os.makedirs(args.output_dir, exist_ok=True)
    pd.Series(label_map).to_json(label_map_file)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, df['label_id'].values,
        test_size=0.2, random_state=42
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'airesearch/wangchanberta-base-att-spm-uncased',
        num_labels=len(label_map)
    )
    
    # Prepare datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}')
        
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            predicted = torch.argmax(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predicted = torch.argmax(outputs.logits, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_safetensors(model, args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f'Model saved to {args.output_dir}')
    
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main()