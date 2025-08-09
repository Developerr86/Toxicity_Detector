import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.layers import TextVectorization

# Configure page
st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model and vectorizer loading
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        # Load the trained model
        model = tf.keras.models.load_model('toxicity.h5')
        
        # Load vectorizer config and weights
        with open('vectorizer_config.pkl', 'rb') as f:
            vectorizer_config = pickle.load(f)
            
        with open('vectorizer_weights.pkl', 'rb') as f:
            vectorizer_weights = pickle.load(f)
        
        # Recreate vectorizer
        vectorizer = TextVectorization.from_config(vectorizer_config)
        vectorizer.set_weights(vectorizer_weights)
        
        # Define toxicity categories
        toxicity_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        return model, vectorizer, toxicity_categories
        
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        return None, None, None

def predict_toxicity(comment, model, vectorizer):
    """Predict toxicity scores for a given comment"""
    try:
        # Vectorize the comment
        vectorized_comment = vectorizer([comment])
        
        # Make prediction
        results = model.predict(vectorized_comment, verbose=0)
        
        return results[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Title and description
    st.title("🛡️ Comment Toxicity Detector")
    st.markdown("""
    This application uses a deep learning model to detect various types of toxicity in text comments.
    The model can identify 6 different categories of toxic content.
    """)
    
    # Load model and vectorizer
    with st.spinner("Loading model and vectorizer..."):
        model, vectorizer, toxicity_categories = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.error("Failed to load the model or vectorizer. Please check if the required files exist.")
        st.info("Required files: 'toxicity.h5' and the training dataset in 'jigsaw-toxic-comment-classification-challenge/train.csv/train.csv'")
        return
    
    st.success("Model loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Model Type:** Bidirectional LSTM
        
        **Toxicity Categories:**
        - 🔴 Toxic
        - 🔥 Severe Toxic
        - 🚫 Obscene
        - ⚠️ Threat
        - 💢 Insult
        - 👥 Identity Hate
        
        **Threshold:** 0.5 (50%)
        """)
        
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. Enter your text in the input box
        2. Click 'Analyze Toxicity'
        3. View the results below
        
        The model will show probability scores and binary predictions for each toxicity category.
        """)
    
    # Main interface
    st.header("💬 Enter Text to Analyze")
    
    # Text input
    user_input = st.text_area(
        "Comment to analyze:",
        placeholder="Enter your comment here...",
        height=100,
        help="Enter any text comment to check for toxicity"
    )
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Toxicity", type="primary", use_container_width=True)
    
    # Example texts
    st.subheader("📝 Try These Examples")
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("😊 Positive Example", use_container_width=True):
            user_input = "Thank you for your helpful contribution to this discussion!"
            st.rerun()
    
    with example_col2:
        if st.button("😠 Negative Example", use_container_width=True):
            user_input = "You are such an idiot and I hate everything you say!"
            st.rerun()
    
    # Perform analysis
    if analyze_button and user_input.strip():
        with st.spinner("Analyzing comment..."):
            predictions = predict_toxicity(user_input, model, vectorizer)
        
        if predictions is not None:
            st.header("📈 Analysis Results")
            
            # Create two columns for results
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.subheader("🎯 Probability Scores")
                
                # Create a dataframe for better visualization
                results_df = pd.DataFrame({
                    'Category': toxicity_categories,
                    'Probability': predictions,
                    'Percentage': [f"{p*100:.1f}%" for p in predictions]
                })
                
                # Display as a styled dataframe
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            with results_col2:
                st.subheader("✅ Binary Predictions (>50%)")
                
                # Binary predictions
                binary_predictions = (predictions > 0.5).astype(int)
                
                for i, category in enumerate(toxicity_categories):
                    is_toxic = binary_predictions[i] == 1
                    probability = predictions[i]
                    
                    if is_toxic:
                        st.error(f"🔴 **{category.upper()}**: Detected ({probability*100:.1f}%)")
                    else:
                        st.success(f"✅ **{category.upper()}**: Not detected ({probability*100:.1f}%)")
            
            # Overall assessment
            st.subheader("🎭 Overall Assessment")
            
            total_toxic_categories = sum(binary_predictions)
            max_probability = max(predictions)
            
            if total_toxic_categories == 0:
                st.success("🎉 **CLEAN**: This comment appears to be non-toxic!")
            elif total_toxic_categories == 1:
                st.warning(f"⚠️ **CAUTION**: This comment shows signs of toxicity in {total_toxic_categories} category.")
            else:
                st.error(f"🚨 **WARNING**: This comment shows signs of toxicity in {total_toxic_categories} categories.")
            
            # Progress bars for visual representation
            st.subheader("📊 Visual Breakdown")
            for i, category in enumerate(toxicity_categories):
                probability = predictions[i]
                # Ensure probability is between 0 and 1
                probability_clamped = max(0.0, min(1.0, float(probability)))
                st.write(f"**{category.title()}**: {probability*100:.1f}%")
                st.progress(probability_clamped)
    
    elif analyze_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Built with Streamlit and TensorFlow | Model trained on Jigsaw Toxic Comment Classification dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
