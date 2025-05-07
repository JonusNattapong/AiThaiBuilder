#!/bin/bash

# Create output directory
mkdir -p models/thai_sentiment

# Run fine-tuning
python ../train/fine_tune_sentiment_model.py \
    --data_path "../../DatasetDownload/processed_sentiment/train.csv" \
    --model_name "airesearch/wangchanberta-base-att-spm-uncased" \
    --output_dir "../../models/thai_sentiment" \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5