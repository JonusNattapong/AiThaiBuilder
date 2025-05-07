#!/bin/bash

# Create necessary directories
mkdir -p ../../data/processed
mkdir -p ../../models/thai_sentiment

# Clean dataset first
python ../clean/clean_sentiment_dataset.py \
    --input_file "../../data/input/sentiment_dataset.csv" \
    --output_file "../../data/processed/cleaned_sentiment_dataset.csv"

# Run fine-tuning
python ../train/fine_tune_sentiment_model.py \
    --data_path "../../data/processed/cleaned_sentiment_dataset.csv" \
    --model_name "airesearch/wangchanberta-base-att-spm-uncased" \
    --output_dir "../../models/thai_sentiment" \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5