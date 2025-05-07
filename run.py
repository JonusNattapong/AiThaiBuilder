import os
import sys
from src.app import demo

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("data", exist_ok=True)
    
    # Launch the Gradio app
    demo.launch(share=False)