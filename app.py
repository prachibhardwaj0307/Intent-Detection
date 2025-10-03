import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Query Intent Classifier",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained BERT model and tokenizer"""
    try:
        # Try to load from the saved model directory from your notebook
        model_paths = [
            "./bert-intent-classifier",  # From your notebook output
            "./results/checkpoint-200",   # Alternative path
            "bert-base-uncased"          # Fallback to base model
        ]

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = BertForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=1,
                        problem_type="single_label_classification"
                    )
                    model.eval()
                    st.success(f"✅ Model loaded successfully from {model_path}")
                    return model, tokenizer, model_path
                except Exception as e:
                    st.warning(f"Failed to load from {model_path}: {str(e)}")
                    continue

        # If no saved model found, use base model with warning
        st.warning("⚠️ Using base BERT model - predictions may not be accurate")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=1,
            problem_type="single_label_classification"
        )
        model.eval()
        return model, tokenizer, "bert-base-uncased (not trained)"

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_intent(query, model, tokenizer, threshold=0.6):
    """Predict intent for a given query"""
    device = torch.device("cpu")

    # Tokenize input
    inputs = tokenizer(
        query, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=64
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probability = torch.sigmoid(outputs.logits).item()
        predicted_label = 1 if probability > threshold else 0

    # Convert to intent labels
    intent = "people" if predicted_label == 1 else "search"
    confidence = probability if predicted_label == 1 else (1 - probability)

    return intent, confidence, probability

def main():
    st.title("🔍 Query Intent Classifier")
    st.markdown("**Classify queries as 'People Search' or 'General Search' using BERT**")
    st.markdown("---")

    # Load model
    model, tokenizer, model_path = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please check the model files.")
        return

    # Model information sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.write(f"**Model:** {model_path}")
        st.write("**Architecture:** BERT (bert-base-uncased)")
        st.write("**Task:** Binary Intent Classification")
        st.write("**Classes:** People Search vs General Search")

        # Threshold adjustment
        threshold = st.slider(
            "Classification Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.6, 
            step=0.05,
            help="Queries with probability above this threshold are classified as 'people'"
        )

    # Main interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Query input
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., 'Find John Smith' or 'Search for restaurants'",
            help="Enter a query to classify as 'people' or 'search' intent"
        )

    with col2:
        st.write("")  # Spacing
        predict_button = st.button("🚀 Classify", type="primary", use_container_width=True)

    # Prediction
    if predict_button or (query and len(query.strip()) > 0):
        if query.strip():
            with st.spinner("🤖 Analyzing query..."):
                try:
                    intent, confidence, raw_probability = predict_intent(
                        query, model, tokenizer, threshold
                    )

                    # Results section
                    st.markdown("### 📊 Results")

                    # Main result display
                    result_col1, result_col2, result_col3 = st.columns([2, 1, 1])

                    with result_col1:
                        if intent == "people":
                            st.success("**Intent:** 👥 People Search")
                        else:
                            st.info("**Intent:** 🔎 General Search")

                    with result_col2:
                        st.metric("Confidence", f"{confidence:.1%}")

                    with result_col3:
                        st.metric("Raw Score", f"{raw_probability:.3f}")

                    # Progress bar for confidence
                    st.progress(confidence)

                    # Detailed information
                    with st.expander("📋 Detailed Analysis"):
                        st.write(f"**Input Query:** `{query}`")
                        st.write(f"**Predicted Intent:** `{intent}`")
                        st.write(f"**Raw Probability:** `{raw_probability:.6f}`")
                        st.write(f"**Classification Threshold:** `{threshold}`")
                        st.write(f"**Confidence Score:** `{confidence:.6f}`")

                        # Decision logic explanation
                        st.markdown("**Decision Logic:**")
                        if raw_probability > threshold:
                            st.write(f"• Raw probability ({raw_probability:.3f}) > threshold ({threshold}) → **People Search**")
                        else:
                            st.write(f"• Raw probability ({raw_probability:.3f}) ≤ threshold ({threshold}) → **General Search**")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query to classify.")

    # Sample queries section
    st.markdown("---")
    st.subheader("💡 Try these sample queries:")

    sample_queries = [
        ("Find John Doe", "👥"),
        ("Search for best restaurants", "🔎"),
        ("Look up Mary Smith contact", "👥"),
        ("Find documents about AI", "🔎"),
        ("Sarah Johnson phone number", "👥"),
        ("Search products online", "🔎"),
        ("Contact information for Dr. Brown", "👥"),
        ("Research about machine learning", "🔎")
    ]

    # Display sample queries in a grid
    cols = st.columns(2)
    for i, (sample, icon) in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(f"{icon} {sample}", key=f"sample_{i}", use_container_width=True):
                # Set the query and trigger prediction
                st.session_state.sample_query = sample
                st.rerun()

    # Handle sample query selection
    if 'sample_query' in st.session_state:
        query = st.session_state.sample_query
        # Clear the session state
        del st.session_state.sample_query

    # Usage statistics (if you want to track)
    if st.checkbox("Show Usage Statistics"):
        st.markdown("### 📈 Session Statistics")
        # You could implement usage tracking here
        st.info("Statistics tracking not implemented yet")

if __name__ == "__main__":
    main()
