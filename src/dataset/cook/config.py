"""
Configuration settings for dataset cooking.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_CACHE_DIR = BASE_DIR / ".model_cache"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Content settings
MAX_LENGTH = 2000
BATCH_SIZE = 10

# Supported languages
SUPPORTED_LANGUAGES = ["EN", "TH", "JA", "KO", "ZH", "VI"]
DEFAULT_LANGUAGE = "EN"

# Writing styles
SUPPORTED_STYLES = [
    "formal", "casual", "academic", "creative", 
    "professional", "conversational", "technical",
    "emotional", "narrative", "descriptive"
]
DEFAULT_STYLE = "formal"

# Supported topics with configurations
SUPPORTED_TOPICS = {
    "advice": {
        "name": "Advice Content",
        "description": "Life advice and guidance",
        "min_length": 100,
        "max_length": 1000,
        "parameters": {"topic_scope": ["personal", "career", "relationship", "health"]}
    },
    "angry": {
        "name": "Angry Content",
        "description": "Expressions of anger and frustration",
        "min_length": 50,
        "max_length": 500,
        "parameters": {"intensity": ["mild", "moderate", "strong"]}
    },
    "entertainment": {
        "name": "Entertainment Content",
        "description": "Entertainment and media content",
        "min_length": 100,
        "max_length": 1500,
        "parameters": {"format": ["review", "news", "interview", "recap"]}
    },
    "environment": {
        "name": "Environmental Content",
        "description": "Environmental topics and issues",
        "min_length": 200,
        "max_length": 2000,
        "parameters": {"focus": ["climate", "conservation", "sustainability"]}
    },
    "happy": {
        "name": "Happy Content",
        "description": "Positive and joyful content",
        "min_length": 50,
        "max_length": 500,
        "parameters": {"theme": ["success", "celebration", "gratitude"]}
    },
    "history": {
        "name": "Historical Content",
        "description": "Historical events and analyses",
        "min_length": 300,
        "max_length": 3000,
        "parameters": {"period": ["ancient", "medieval", "modern", "contemporary"]}
    },
    "philosophy": {
        "name": "Philosophical Content",
        "description": "Philosophical discussions and thoughts",
        "min_length": 300,
        "max_length": 3000,
        "parameters": {"branch": ["ethics", "metaphysics", "epistemology", "logic"]}
    },
    "politics": {
        "name": "Political Content",
        "description": "Political topics and discussions",
        "min_length": 200,
        "max_length": 2000,
        "parameters": {"aspect": ["policy", "governance", "international", "domestic"]}
    },
    "sad": {
        "name": "Sad Content",
        "description": "Melancholic and emotional content",
        "min_length": 50,
        "max_length": 500,
        "parameters": {"theme": ["loss", "disappointment", "longing"]}
    },
    "technology": {
        "name": "Technology Content",
        "description": "Tech articles and guides",
        "min_length": 200,
        "max_length": 2000,
        "parameters": {"content_type": ["tutorial", "review", "analysis", "news"]}
    },
    "travel": {
        "name": "Travel Content",
        "description": "Travel guides and experiences",
        "min_length": 200,
        "max_length": 2000,
        "parameters": {"type": ["guide", "review", "experience", "tips"]}
    }
}

# Model configuration
MODEL_CONFIG = {
    "provider": "huggingface",
    "model_name": "gpt2-large",
    "temperature": 0.7,
    "max_tokens": 1000,
    "device": "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu",
    "cache_dir": MODEL_CACHE_DIR
}

# Prompt templates
PROMPT_TEMPLATES = {
    "base": """
Generate {style} content for {topic} category.
Topic parameters: {params}
Target length: {min_length}-{max_length} words.
Language: {language}

Content should match the following requirements:
- Appropriate tone for the topic
- Coherent structure
- Natural language flow
- Relevant details and examples
- Topic-appropriate terminology
    """,
    
    "styles": {
        "formal": "Use professional and structured language.",
        "casual": "Write in a relaxed, conversational tone.",
        "academic": "Employ scholarly language and cite sources.",
        "creative": "Use expressive and imaginative language.",
        "professional": "Maintain business-appropriate tone.",
        "conversational": "Write as if speaking to a friend.",
        "technical": "Use precise technical terminology.",
        "emotional": "Express feelings and personal perspectives.",
        "narrative": "Tell a story with clear progression.",
        "descriptive": "Use rich, detailed descriptions."
    }
}

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_FILENAME = "content_generation.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Create log directory
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
VISUALIZATION = {
    "max_examples": 5,
    "chart_height": 400,
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}

# Cache settings
ENABLE_CACHE = True
CACHE_DIR = BASE_DIR / ".cache"
CACHE_TTL = 24 * 60 * 60  # 24 hours
MAX_CACHE_SIZE = 1000  # entries

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Metadata analysis settings
METADATA_CONFIG = {
    "min_confidence": 0.7,
    "max_categories": 5,
    "sentiment_analysis": True,
    "extract_keywords": True,
    "keyword_limit": 10,
    "save_metrics": True
}

# Topic-specific metadata fields
METADATA_FIELDS = {
    "common": ["language", "style", "word_count", "sentiment", "keywords"],
    "topic_specific": {
        "advice": ["category", "target_audience", "difficulty"],
        "angry": ["intensity", "trigger", "resolution"],
        "entertainment": ["genre", "medium", "target_age"],
        "environment": ["issue_type", "geographic_scope", "urgency"],
        "happy": ["occasion", "mood_intensity", "celebration_type"],
        "history": ["era", "region", "historical_figures"],
        "philosophy": ["school_of_thought", "key_concepts", "philosophers"],
        "politics": ["political_level", "ideology", "region"],
        "sad": ["emotion_type", "trigger", "resolution"],
        "technology": ["tech_domain", "complexity", "timeframe"],
        "travel": ["destination", "activity_type", "season"]
    }
}