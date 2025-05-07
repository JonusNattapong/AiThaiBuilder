# Dataset Cook

Interactive tool for generating content datasets using language models with Streamlit interface.

## Features

- Multiple content topics supported:
  - Advice Content
  - Technology Content
  - Entertainment Content

- Content customization:
  - Multiple languages (EN, TH, JA, KO, ZH, VI)
  - Writing styles (formal, casual, academic, etc.)
  - Topic-specific parameters
  - Length control

- Interactive interface:
  - Real-time content generation
  - Content preview and editing
  - Metadata management
  - Multiple export formats

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DatasetCook
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download language models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download th_core_news_lg
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Using the interface:
   - Select content topic
   - Choose language and style
   - Configure parameters
   - Generate content
   - Review and edit
   - Export dataset

## Supported Topics

### Advice Content
- Parameters:
  - Topic scope (personal, career, relationship, health)
- Metadata:
  - Category
  - Difficulty level

### Technology Content
- Parameters:
  - Content type (tutorial, review, analysis, news)
- Metadata:
  - Tech category
  - Expertise level

### Entertainment Content
- Parameters:
  - Format (review, news, interview, recap)
- Metadata:
  - Genre
  - Target audience

## Writing Styles

- Formal
- Casual
- Academic
- Creative
- Professional
- Conversational
- Technical

## Output Formats

### JSON
```json
{
  "text": "Generated content...",
  "metadata": {
    "category": "personal",
    "difficulty": "intermediate"
  },
  "language": "EN",
  "style": "formal",
  "topic": "advice"
}
```

### CSV
Contains flattened version of JSON for easy analysis.

### JSONL
Line-by-line JSON format for large datasets.

## Configuration

Edit `config.py` to modify:
- Topic definitions
- Language models
- Content parameters
- Output settings
- Cache configuration

## Directory Structure

```
DatasetCook/
├── streamlit_app.py    # Streamlit interface
├── cook_utils.py      # Content generation utilities
├── config.py         # Configuration settings
├── requirements.txt  # Dependencies
├── output/          # Generated datasets
├── .model_cache/    # Cached models
└── logs/           # Application logs
```

## Model Support

- Local models via HuggingFace
- OpenAI API (optional)
- Custom model integration

## Content Validation

Validates:
- Content length
- Required metadata
- Topic-specific rules
- Language consistency

## Metadata Generation

Automatically extracts:
- Content categories
- Keywords
- Sentiment
- Complexity metrics

## Cache System

- Model caching
- Content caching
- Configurable TTL
- Size limits

## Error Handling

- Invalid configurations
- Generation failures
- Model errors
- File operations

## Contributing

Feel free to submit issues and pull requests for:
- New topics
- Content templates
- Language support
- Interface improvements
- Bug fixes

## License

MIT License