# RunThaiGenDataset - Thai Dataset Generation Tool 🤖

RunThaiGenDataset is a powerful and flexible tool designed to generate high-quality Thai language datasets for various Natural Language Processing (NLP) tasks. It leverages the capabilities of the Deepseek API for text generation and provides a user-friendly Gradio interface for easy interaction.

## ✨ Features

-   **Versatile Dataset Generation:** Supports a wide range of NLP tasks including:
    -   Text Generation
    -   Summarization
    -   Question Answering
    -   Translation (EN-TH, ZH-TH)
    -   Instruction Following
    -   And many more! (Refer to `config/config.json` for a full list)
-   **Powered by Deepseek API:** Utilizes state-of-the-art language models for generating coherent and contextually relevant Thai text.
-   **Customizable Prompts:** Allows users to define custom system prompts and additional instructions to tailor the generation process.
-   **User-Friendly Interface:** Built with Gradio for an intuitive web-based experience, making dataset creation accessible to everyone.
-   **Batch Generation:** Easily generate multiple samples for your datasets.
-   **Organized Output:** Saves generated datasets in structured formats, ready for fine-tuning models or other NLP pipelines.
-   **Extensible:** Designed with a modular configuration, making it easy to add new tasks or modify existing ones.

## 🛠️ Setup

Follow these steps to get RunThaiGenDataset up and running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/RunThaiGenDataset.git
    cd RunThaiGenDataset
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
    Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Deepseek API Key:**
    Obtain your Deepseek API key from [https://platform.deepseek.com/](https://platform.deepseek.com/).
    Create a `.env` file in the project root directory (`RunThaiGenDataset/.env`) and add your API key:
    ```env
    # .env
    DEEPSEEK_API_KEY="your_actual_deepseek_api_key_here"
    ```
    Alternatively, you can enter the API key directly in the Gradio UI when running the application.

## 🚀 Running the Application

Once the setup is complete, you can start the Gradio application:

```bash
python src/app.py
```

Navigate to the local URL provided in your terminal (usually `http://127.0.0.1:7860`) using your web browser to access the Thai Dataset Generator.

## 📖 API Documentation (Conceptual)

This project includes an OpenAPI specification (`openapi.yaml`) that describes the conceptual API for the dataset generation functionality. This is useful for understanding the data structures and parameters involved.

You can view this documentation using **ReDoc**.

### Viewing with ReDoc:

1.  **Using a Docker container (Recommended):**
    If you have Docker installed, you can quickly serve the ReDoc interface. Navigate to your project directory in the terminal:
    ```bash
    # For PowerShell:
    # docker run -p 8080:80 -v ${PWD}\openapi.yaml:/usr/share/nginx/html/openapi.yaml redocly/redoc

    # For CMD (ensure you are in the project directory):
    # docker run -p 8080:80 -v "%CD%\openapi.yaml":/usr/share/nginx/html/openapi.yaml redocly/redoc

    # For Git Bash or Linux-like terminals:
    # docker run -p 8080:80 -v $(pwd)/openapi.yaml:/usr/share/nginx/html/openapi.yaml redocly/redoc
    ```
    Then open `http://localhost:8080` in your browser. It should automatically load `openapi.yaml`. If not, try `http://localhost:8080/?url=openapi.yaml`.

2.  **Using an online viewer:**
    You can use an online ReDoc viewer. Upload or paste the content of `openapi.yaml` into such a viewer.
    Example: [https://redocly.github.io/redoc/](https://redocly.github.io/redoc/)
    *(Note: Be cautious about pasting sensitive information or proprietary schemas into public online tools).*

3.  **Using `redoc-cli` (Node.js required):**
    If you have Node.js and npm installed:
    ```bash
    # Install redoc-cli globally (only needs to be done once)
    npm install -g redoc-cli

    # Navigate to your project directory
    # cd path/to/RunThaiGenDataset

    # Serve the documentation
    redoc-cli serve openapi.yaml
    ```
    This will typically open the documentation in your browser at `http://127.0.0.1:8080/`.

## 🧑‍💻 Developers

This project is developed and maintained by:

-   **JonusNattapong**
-   **zombitx64**

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable, otherwise specify).