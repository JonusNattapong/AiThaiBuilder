# RunThaiGenDataset - Thai Dataset Generation Tool

This project provides a Gradio interface to generate Thai datasets for various NLP tasks using the Deepseek API.

## Features

-   Supports multiple NLP tasks.
-   Uses Deepseek API for text generation.
-   Allows customization of prompts and number of samples.
-   Outputs datasets for fine-tuning models.

## Setup

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Obtain a Deepseek API key from [https://platform.deepseek.com/](https://platform.deepseek.com/) and place it in a `.env` file in the project root:
    ```env
    # .env
    DEEPSEEK_API_KEY="your_actual_deepseek_api_key_here"
    ```
    Alternatively, you can enter the API key directly in the Gradio UI.

## Running the Application

```bash
python app.py
```

Navigate to the local URL provided by Gradio in your web browser.

## API Documentation (Conceptual)

This project includes an OpenAPI specification (`openapi.yaml`) that describes the conceptual API for the dataset generation functionality. You can view this documentation using ReDoc.

**Viewing with ReDoc:**

1.  **Using a Docker container (Recommended):**
    If you have Docker installed, you can quickly serve the ReDoc interface:
    ```bash
    # Navigate to your project directory in the terminal
    # cd c:\Users\Admin\OneDrive\เอกสาร\Github\RunThaiGenDataset

    # For PowerShell:
    # docker run -p 8080:80 -v ${PWD}\openapi.yaml:/usr/share/nginx/html/openapi.yaml redocly/redoc
    # For CMD (ensure you are in the project directory):
    # docker run -p 8080:80 -v "%CD%\openapi.yaml":/usr/share/nginx/html/openapi.yaml redocly/redoc
    # For Git Bash or Linux-like terminals:
    # docker run -p 8080:80 -v $(pwd)/openapi.yaml:/usr/share/nginx/html/openapi.yaml redocly/redoc
    ```
    Then open `http://localhost:8080` in your browser. It should automatically load `openapi.yaml`.
    If it doesn't load automatically, try `http://localhost:8080/?url=openapi.yaml`.

2.  **Using an online viewer:**
    You can use an online ReDoc viewer. Upload or paste the content of `openapi.yaml` into such a viewer.
    Example: [https://redocly.github.io/redoc/](https://redocly.github.io/redoc/) (Note: be cautious about pasting sensitive information or proprietary schemas into public online tools).

3.  **Using `redoc-cli` (Node.js required):**
    If you have Node.js and npm installed:
    ```bash
    # Install redoc-cli globally (only needs to be done once)
    npm install -g redoc-cli

    # Navigate to your project directory
    # cd c:\Users\Admin\OneDrive\เอกสาร\Github\RunThaiGenDataset

    # Serve the documentation
    redoc-cli serve openapi.yaml
    ```
    This will typically open the documentation in your browser at `http://127.0.0.1:8080/`.

This documentation helps understand the parameters involved in generating datasets with this tool.