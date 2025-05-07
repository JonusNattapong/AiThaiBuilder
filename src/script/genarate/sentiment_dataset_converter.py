import json
import pandas as pd
import os

def convert_json_to_csv():
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to project root
    input_path = os.path.join('output', 'sentiment_dataset_110.json')
    output_dir = os.path.join('data', 'processed')
    
    print(f"Reading from: {input_path}")
    
    # Read JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'text': data,
        'label': ['sentiment'] * len(data)  # Default label for now
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    output_path = os.path.join(output_dir, 'sentiment_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    convert_json_to_csv()