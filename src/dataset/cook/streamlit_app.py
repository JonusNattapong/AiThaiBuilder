import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from cook_utils import (
    CookManager,
    load_dataset,
    save_dataset,
    get_supported_topics,
    get_supported_languages,
    validate_dataset
)
from config import (
    OUTPUT_DIR, BATCH_SIZE, MAX_LENGTH,
    SUPPORTED_TOPICS, DEFAULT_TOPIC,
    SUPPORTED_STYLES, DEFAULT_STYLE
)

def init_session_state():
    """Initialize session state variables."""
    if 'cook_manager' not in st.session_state:
        st.session_state.cook_manager = None
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'edited_results' not in st.session_state:
        st.session_state.edited_results = {}

def display_content_preview(content: dict, index: int):
    """Display content preview with editing capability."""
    st.subheader(f"Content {index + 1}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if "title" in content:
            edited_title = st.text_input(
                "Title",
                st.session_state.edited_results.get(f"{index}_title", content["title"]),
                key=f"title_{index}"
            )
            st.session_state.edited_results[f"{index}_title"] = edited_title

        edited_text = st.text_area(
            "Content",
            st.session_state.edited_results.get(f"{index}_text", content["text"]),
            height=200,
            key=f"text_{index}"
        )
        st.session_state.edited_results[f"{index}_text"] = edited_text

    with col2:
        if "metadata" in content:
            st.write("Metadata:")
            for key, value in content["metadata"].items():
                edited_value = st.text_input(
                    key,
                    st.session_state.edited_results.get(f"{index}_meta_{key}", value),
                    key=f"meta_{index}_{key}"
                )
                st.session_state.edited_results[f"{index}_meta_{key}"] = edited_value

def main():
    st.set_page_config(page_title="Dataset Cook", layout="wide")
    st.title("Dataset Cook")

    # Initialize state
    init_session_state()

    # Sidebar configuration
    st.sidebar.header("Content Configuration")
    
    # Topic selection 
    topics = get_supported_topics()
    topic = st.sidebar.selectbox(
        "Select Topic",
        list(topics.keys()),
        format_func=lambda x: topics[x]["name"]
    )

    if topic != st.session_state.current_topic:
        st.session_state.current_topic = topic
        st.session_state.cook_manager = CookManager(topic)
        st.session_state.results = []
        st.session_state.edited_results = {}

    # Language selection
    languages = get_supported_languages()
    language = st.sidebar.selectbox(
        "Language",
        languages
    )

    # Writing style
    style = st.sidebar.selectbox(
        "Writing Style",
        SUPPORTED_STYLES,
        index=SUPPORTED_STYLES.index(DEFAULT_STYLE)
    )

    # Topic-specific settings
    topic_config = {}
    if topic in topics:
        st.sidebar.subheader("Topic Settings")
        for param, details in topics[topic].get("parameters", {}).items():
            if details["type"] == "number":
                topic_config[param] = st.sidebar.number_input(
                    details["name"],
                    min_value=details.get("min", 0),
                    max_value=details.get("max", 1000),
                    value=details.get("default", 0)
                )
            elif details["type"] == "select":
                topic_config[param] = st.sidebar.selectbox(
                    details["name"],
                    options=details["options"],
                    index=details["options"].index(details.get("default"))
                )
            elif details["type"] == "boolean":
                topic_config[param] = st.sidebar.checkbox(
                    details["name"],
                    value=details.get("default", False)
                )

    # Main content
    tab1, tab2, tab3 = st.tabs(["Generate", "Review", "Export"])

    with tab1:
        st.header("Content Generation")

        # Generation settings
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.number_input(
                "Number of Samples",
                min_value=1,
                max_value=1000,
                value=10
            )
        with col2:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=num_samples,
                value=min(BATCH_SIZE, num_samples)
            )

        if st.button("Generate Content"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Generate content
                results = st.session_state.cook_manager.generate_content(
                    num_samples=num_samples,
                    language=language,
                    style=style,
                    config=topic_config,
                    progress_callback=lambda p: progress_bar.progress(p)
                )
                
                st.session_state.results = results
                st.success("Content generation completed!")
                
                # Display preview
                st.subheader("Preview")
                st.dataframe(pd.DataFrame(results))

            except Exception as e:
                st.error(f"Error generating content: {str(e)}")

    with tab2:
        st.header("Content Review")
        
        if not st.session_state.results:
            st.info("No content to review. Generate content first.")
            return

        # Display all content pieces
        for idx, content in enumerate(st.session_state.results):
            display_content_preview(content, idx)
            st.markdown("---")

    with tab3:
        st.header("Export Content")
        
        if not st.session_state.results:
            st.info("No content to export. Generate content first.")
            return

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            format_type = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "JSONL"]
            )
        with col2:
            filename = st.text_input("Filename", f"{topic}_dataset")

        if st.button("Export Dataset"):
            try:
                # Prepare export data with edited content
                export_data = []
                for idx, content in enumerate(st.session_state.results):
                    updated_content = content.copy()
                    
                    # Update with edited values
                    if f"{idx}_title" in st.session_state.edited_results:
                        updated_content["title"] = st.session_state.edited_results[f"{idx}_title"]
                    if f"{idx}_text" in st.session_state.edited_results:
                        updated_content["text"] = st.session_state.edited_results[f"{idx}_text"]
                    
                    # Update metadata
                    if "metadata" in content:
                        for key in content["metadata"]:
                            meta_key = f"{idx}_meta_{key}"
                            if meta_key in st.session_state.edited_results:
                                updated_content["metadata"][key] = st.session_state.edited_results[meta_key]
                    
                    export_data.append(updated_content)

                # Save dataset
                output_path = save_dataset(
                    export_data,
                    filename,
                    format_type,
                    OUTPUT_DIR
                )
                st.success(f"Dataset exported to: {output_path}")

            except Exception as e:
                st.error(f"Error exporting dataset: {str(e)}")

if __name__ == "__main__":
    main()