"""
Utility functions for dataset cooking and content generation.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from pathlib import Path
from transformers import pipeline
import openai
from config import (
    SUPPORTED_TOPICS, SUPPORTED_LANGUAGES, 
    SUPPORTED_STYLES, MODEL_CONFIG, 
    PROMPT_TEMPLATES
)

class CookManager:
    """Manages content generation and processing."""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.topic_config = SUPPORTED_TOPICS[topic]
        self.generator = pipeline(
            "text-generation",
            model=MODEL_CONFIG["model_name"],
            device=MODEL_CONFIG["device"]
        )

    def generate_content(self,
                        num_samples: int,
                        language: str,
                        style: str,
                        config: Dict[str, Any],
                        progress_callback: Callable = None) -> List[Dict]:
        """Generate content based on topic and configuration."""
        results = []
        prompt_template = PROMPT_TEMPLATES[self.topic]

        for i in range(num_samples):
            try:
                # Generate main content
                prompt = self._prepare_prompt(
                    prompt_template,
                    language,
                    style,
                    config
                )
                
                generated_text = self.generator(
                    prompt,
                    max_new_tokens=MODEL_CONFIG["max_tokens"],
                    temperature=MODEL_CONFIG["temperature"],
                    num_return_sequences=1
                )[0]["generated_text"]

                # Parse and structure content
                content = self._parse_content(generated_text, config)
                
                # Generate metadata
                metadata = self._generate_metadata(content, config)
                
                results.append({
                    "text": content,
                    "metadata": metadata,
                    "language": language,
                    "style": style,
                    "topic": self.topic
                })

                if progress_callback:
                    progress_callback((i + 1) / num_samples)

            except Exception as e:
                print(f"Error generating content {i+1}: {str(e)}")
                continue

        return results

    def _prepare_prompt(self, template: str, language: str, style: str, config: Dict) -> str:
        """Prepare generation prompt with parameters."""
        # Replace placeholders in template
        prompt = template.format(
            language=language,
            style=style,
            **config
        )
        
        # Add style instructions
        style_instructions = PROMPT_TEMPLATES["styles"][style]
        prompt += f"\n\n{style_instructions}"
        
        # Add language instructions if not English
        if language != "EN":
            prompt += f"\n\nPlease write in {language}."
            
        return prompt

    def _parse_content(self, text: str, config: Dict) -> str:
        """Parse and clean generated content."""
        # Basic cleaning
        text = text.strip()
        
        # Remove prompt artifacts
        if ":" in text:
            text = text.split(":", 1)[1].strip()
            
        # Apply topic-specific parsing
        if self.topic_config.get("parse_function"):
            text = self.topic_config["parse_function"](text)
            
        return text

    def _generate_metadata(self, content: str, config: Dict) -> Dict:
        """Generate metadata for content."""
        metadata = {
            "length": len(content),
            "complexity": self._calculate_complexity(content)
        }
        
        # Add topic-specific metadata
        if self.topic_config.get("metadata_function"):
            topic_metadata = self.topic_config["metadata_function"](content)
            metadata.update(topic_metadata)
            
        return metadata

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count == 0:
            sentence_count = 1
            
        words_per_sentence = len(words) / sentence_count
        
        return (avg_word_length * 0.5 + words_per_sentence * 0.5)

def get_supported_topics() -> Dict[str, Dict]:
    """Get dictionary of supported topics."""
    return SUPPORTED_TOPICS

def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif ext == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif ext == '.csv':
        return pd.read_csv(file_path).to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_dataset(data: List[Dict],
                filename: str,
                format_type: str,
                output_dir: str) -> str:
    """Save dataset in specified format."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(
        output_dir,
        f"{base_filename}.{format_type.lower()}"
    )
    
    try:
        if format_type.upper() == "JSON":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format_type.upper() == "JSONL":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format_type.upper() == "CSV":
            pd.DataFrame(data).to_csv(output_path, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {str(e)}")

def validate_dataset(data: List[Dict],
                    topic: str,
                    config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset content and structure."""
    issues = []
    topic_config = SUPPORTED_TOPICS[topic]
    
    for i, item in enumerate(data):
        # Check required fields
        if "text" not in item:
            issues.append(f"Item {i}: Missing 'text' field")
        if "metadata" not in item:
            issues.append(f"Item {i}: Missing 'metadata' field")
            
        # Check content length
        if len(item.get("text", "")) < topic_config.get("min_length", 0):
            issues.append(f"Item {i}: Content too short")
        if len(item.get("text", "")) > topic_config.get("max_length", float('inf')):
            issues.append(f"Item {i}: Content too long")
            
        # Check metadata fields
        required_metadata = topic_config.get("required_metadata", [])
        for field in required_metadata:
            if field not in item.get("metadata", {}):
                issues.append(f"Item {i}: Missing required metadata '{field}'")
                
        # Topic-specific validation
        if topic_config.get("validate_function"):
            topic_issues = topic_config["validate_function"](item)
            issues.extend(f"Item {i}: {issue}" for issue in topic_issues)
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }

def setup_language_model() -> None:
    """Setup and configure language model."""
    # Set API keys if using OpenAI
    if MODEL_CONFIG["provider"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Load local models if specified
    if MODEL_CONFIG["provider"] == "local":
        os.environ["CUDA_VISIBLE_DEVICES"] = MODEL_CONFIG.get("gpu_devices", "0")
        
    # Set cache directory
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CONFIG["cache_dir"]