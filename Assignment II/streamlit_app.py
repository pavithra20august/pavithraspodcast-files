"""
Unified Streamlit Application for Feedback Loop NLP Model
This combines all functionality into one comprehensive web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
import logging

# Add the current directory to path to import our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Feedback Loop AI Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model with caching"""
    try:
        from feedback_nlp_model import FeedbackLoopNLPModel
        
        model = FeedbackLoopNLPModel(
            model_name="distilbert-base-uncased",
            num_labels=2,
            learning_rate=2e-5,
            feedback_weight=0.3
        )
        
        # Try to load existing model
        if os.path.exists("simple_demo_model.pth"):
            model.load_model_state("simple_demo_model.pth")
            st.success("✅ Loaded existing trained model")
        else:
            st.warning("⚠️ Using untrained model (predictions may not be accurate)")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'model_stats' not in st.session_state:
        st.session_state.model_stats = {}

def main():
    """Main application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Feedback Loop AI Model</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced AI model that learns and improves through user feedback
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if st.session_state.model is None:
        with st.spinner('Loading AI model...'):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Failed to load model. Please check the error messages above.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📝 Text Prediction", "🔄 Feedback Collection", "📊 Analytics", "🏋️ Model Training", "⚙️ Settings"]
    )
    
    # Route to different pages
    if page == "🏠 Home":
        home_page()
    elif page == "📝 Text Prediction":
        prediction_page()
    elif page == "🔄 Feedback Collection":
        feedback_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "🏋️ Model Training":
        training_page()
    elif page == "⚙️ Settings":
        settings_page()

def home_page():
    """Home page with overview and quick actions"""
    st.header("🏠 Welcome to the Feedback Loop AI Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is this?
        This is an advanced AI model that can analyze text sentiment and continuously improve through user feedback.
        
        ### ✨ Key Features:
        - **Real-time Sentiment Analysis**: Get instant predictions on text sentiment
        - **Feedback Loop Learning**: The model learns from your corrections
        - **Confidence Scoring**: See how confident the model is in its predictions
        - **Performance Analytics**: Track model improvement over time
        - **Multiple Interfaces**: Use command-line, interactive, or web interface
        
        ### 🚀 Quick Start:
        1. Go to **Text Prediction** to analyze sentiment
        2. Provide feedback on predictions in **Feedback Collection**
        3. View performance in **Analytics**
        4. Retrain the model in **Model Training**
        """)
    
    with col2:
        # Model status
        st.markdown("### 📊 Model Status")
        
        if hasattr(st.session_state.model, 'feedback_history'):
            feedback_count = len(st.session_state.model.feedback_history)
            if feedback_count > 0:
                avg_score = np.mean([f['feedback_score'] for f in st.session_state.model.feedback_history])
                st.metric("Feedback Samples", feedback_count)
                st.metric("Avg Feedback Score", f"{avg_score:.2f}")
            else:
                st.info("No feedback collected yet")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🎯 Test Sample Text", use_container_width=True):
            st.session_state.test_text = "I love this AI model!"
            st.rerun()
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "📊 Analytics"
            st.rerun()
        
        if st.button("🔄 Provide Feedback", use_container_width=True):
            st.session_state.page = "🔄 Feedback Collection"
            st.rerun()

def prediction_page():
    """Text prediction page"""
    st.header("📝 Text Sentiment Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        st.subheader("Enter your text:")
        
        # Check if there's a test text from home page
        default_text = st.session_state.get('test_text', '')
        if default_text:
            user_text = st.text_area(
                "Type your text here:",
                value=default_text,
                height=100,
                placeholder="Enter a sentence or paragraph for sentiment analysis..."
            )
            # Clear the test text after using it
            if 'test_text' in st.session_state:
                del st.session_state.test_text
        else:
            user_text = st.text_area(
                "Type your text here:",
                height=100,
                placeholder="Enter a sentence or paragraph for sentiment analysis..."
            )
        
        # Prediction button
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing..."):
                    try:
                        pred_label, confidence, details = st.session_state.model.predict_with_confidence(user_text)
                        
                        # Display results
                        sentiment = "Positive" if pred_label == 1 else "Negative"
                        confidence_pct = confidence * 100
                        
                        # Color coding
                        if pred_label == 1:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>🎉 Sentiment: {sentiment}</h3>
                                <p><strong>Confidence:</strong> {confidence_pct:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                <h3>😞 Sentiment: {sentiment}</h3>
                                <p><strong>Confidence:</strong> {confidence_pct:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(confidence)
                        
                        # Detailed probabilities
                        with st.expander("📊 Detailed Analysis"):
                            prob_data = {
                                'Sentiment': ['Negative', 'Positive'],
                                'Probability': details['probabilities']
                            }
                            prob_df = pd.DataFrame(prob_data)
                            
                            fig = px.bar(
                                prob_df, 
                                x='Sentiment', 
                                y='Probability',
                                title="Prediction Probabilities",
                                color='Sentiment',
                                color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#51cf66'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Store prediction for feedback
                        st.session_state.last_prediction = {
                            'text': user_text,
                            'predicted_label': pred_label,
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        }
                        
                        # Add to prediction history
                        st.session_state.prediction_history.append({
                            'text': user_text,
                            'sentiment': sentiment,
                            'confidence': confidence_pct,
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        # Model statistics
        st.subheader("📈 Model Statistics")
        
        if hasattr(st.session_state.model, 'feedback_history') and st.session_state.model.feedback_history:
            total_feedback = len(st.session_state.model.feedback_history)
            avg_feedback_score = np.mean([f['feedback_score'] for f in st.session_state.model.feedback_history])
            
            st.metric("Total Feedback", total_feedback)
            st.metric("Avg Feedback Score", f"{avg_feedback_score:.2f}")
            
            # Recent feedback
            recent_feedback = list(st.session_state.model.feedback_history)[-5:]
            st.subheader("Recent Feedback")
            for i, feedback in enumerate(recent_feedback):
                score = feedback['feedback_score']
                color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                st.write(f"{color} Score: {score:.2f}")
        else:
            st.info("No feedback collected yet.")
        
        # Recent predictions
        if st.session_state.prediction_history:
            st.subheader("Recent Predictions")
            for pred in st.session_state.prediction_history[-5:]:
                st.write(f"📝 {pred['sentiment']} ({pred['confidence']:.1f}%)")

def feedback_page():
    """Feedback collection page"""
    st.header("💬 Provide Feedback")
    
    if 'last_prediction' not in st.session_state:
        st.warning("Please make a prediction first on the Text Prediction page.")
        return
    
    last_pred = st.session_state.last_prediction
    
    # Display last prediction
    st.subheader("Your Last Prediction:")
    st.write(f"**Text:** {last_pred['text']}")
    st.write(f"**Predicted Sentiment:** {'Positive' if last_pred['predicted_label'] == 1 else 'Negative'}")
    st.write(f"**Confidence:** {last_pred['confidence']:.3f}")
    
    # Feedback form
    st.subheader("Was the prediction correct?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Correct", type="primary", use_container_width=True):
            actual_label = last_pred['predicted_label']
            user_rating = st.slider("Rate the prediction quality:", 1.0, 5.0, 4.0, 0.1)
            
            st.session_state.model.add_feedback(
                last_pred['text'],
                last_pred['predicted_label'],
                actual_label,
                last_pred['confidence'],
                user_rating
            )
            
            st.success("✅ Feedback recorded! Thank you for helping improve the model.")
            del st.session_state.last_prediction
            st.rerun()
    
    with col2:
        if st.button("❌ Incorrect", use_container_width=True):
            actual_label = 1 - last_pred['predicted_label']  # Flip the label
            user_rating = st.slider("Rate the prediction quality:", 1.0, 5.0, 2.0, 0.1)
            
            st.session_state.model.add_feedback(
                last_pred['text'],
                last_pred['predicted_label'],
                actual_label,
                last_pred['confidence'],
                user_rating
            )
            
            st.error("❌ Feedback recorded! The model will learn from this mistake.")
            del st.session_state.last_prediction
            st.rerun()
    
    # Manual feedback section
    st.subheader("📝 Manual Feedback")
    
    manual_text = st.text_area("Enter text for manual feedback:")
    manual_sentiment = st.selectbox("Actual sentiment:", ["Positive", "Negative"])
    manual_rating = st.slider("Your rating:", 1.0, 5.0, 3.0, 0.1)
    
    if st.button("Submit Manual Feedback"):
        if manual_text.strip():
            actual_label = 1 if manual_sentiment == "Positive" else 0
            pred_label, confidence, _ = st.session_state.model.predict_with_confidence(manual_text)
            
            st.session_state.model.add_feedback(
                manual_text,
                pred_label,
                actual_label,
                confidence,
                manual_rating
            )
            
            st.success("Manual feedback submitted successfully!")
            st.rerun()
        else:
            st.warning("Please enter text for feedback.")

def analytics_page():
    """Analytics and performance page"""
    st.header("📊 Performance Analytics")
    
    if not hasattr(st.session_state.model, 'feedback_history') or not st.session_state.model.feedback_history:
        st.info("No feedback data available yet. Please provide some feedback first.")
        return
    
    # Convert feedback history to DataFrame
    feedback_data = []
    for feedback in st.session_state.model.feedback_history:
        feedback_data.append({
            'Timestamp': feedback['timestamp'],
            'Text': feedback['text'][:50] + "..." if len(feedback['text']) > 50 else feedback['text'],
            'Predicted': 'Positive' if feedback['predicted_label'] == 1 else 'Negative',
            'Actual': 'Positive' if feedback['actual_label'] == 1 else 'Negative',
            'Confidence': feedback['confidence'],
            'Feedback Score': feedback['feedback_score'],
            'User Rating': feedback['user_rating'],
            'Error': feedback['error']
        })
    
    feedback_df = pd.DataFrame(feedback_data)
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", len(feedback_df))
    
    with col2:
        accuracy = 1 - feedback_df['Error'].mean()
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col3:
        avg_confidence = feedback_df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col4:
        avg_feedback_score = feedback_df['Feedback Score'].mean()
        st.metric("Avg Feedback Score", f"{avg_feedback_score:.3f}")
    
    # Visualizations
    st.subheader("📈 Performance Trends")
    
    # Feedback score over time
    fig1 = px.line(
        feedback_df, 
        x='Timestamp', 
        y='Feedback Score',
        title="Feedback Score Over Time",
        markers=True
    )
    fig1.update_layout(xaxis_title="Time", yaxis_title="Feedback Score")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Confidence vs Accuracy
    fig2 = px.scatter(
        feedback_df,
        x='Confidence',
        y='Feedback Score',
        color='Error',
        title="Confidence vs Feedback Score",
        color_discrete_map={0: 'green', 1: 'red'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.histogram(
            feedback_df,
            x='Feedback Score',
            title="Feedback Score Distribution",
            nbins=20
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.histogram(
            feedback_df,
            x='Confidence',
            title="Confidence Distribution",
            nbins=20
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Recent feedback table
    st.subheader("📋 Recent Feedback")
    st.dataframe(
        feedback_df.tail(10),
        use_container_width=True,
        hide_index=True
    )

def training_page():
    """Model training page"""
    st.header("🏋️ Model Training")
    
    st.subheader("Current Model Status")
    
    if hasattr(st.session_state.model, 'feedback_history'):
        feedback_count = len(st.session_state.model.feedback_history)
        st.write(f"**Feedback samples collected:** {feedback_count}")
        
        if feedback_count >= 5:
            st.success("✅ Sufficient feedback for retraining!")
            
            if st.button("🔄 Retrain Model with Feedback", type="primary"):
                with st.spinner("Retraining model with feedback data..."):
                    st.session_state.model.retrain_with_feedback(min_feedback_samples=5)
                    st.success("Model retrained successfully!")
                    
                    # Clear feedback history after retraining
                    st.session_state.model.feedback_history.clear()
                    st.info("Feedback history cleared after retraining.")
                    st.rerun()
        else:
            st.warning(f"⚠️ Need at least 5 feedback samples for retraining. Current: {feedback_count}")
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Architecture:**")
        st.write(f"- Base Model: {st.session_state.model.model_name}")
        st.write(f"- Number of Labels: {st.session_state.model.num_labels}")
        st.write(f"- Learning Rate: {st.session_state.model.learning_rate}")
    
    with col2:
        st.write("**Feedback System:**")
        st.write(f"- Feedback Weight: {st.session_state.model.feedback_weight}")
        st.write(f"- Max Feedback History: {st.session_state.model.max_feedback_history}")
        st.write(f"- Device: {st.session_state.model.device}")
    
    st.subheader("Model Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Model"):
            st.session_state.model.save_model_state("streamlit_model.pth")
            st.success("Model saved successfully!")
    
    with col2:
        if st.button("📊 Generate Performance Report"):
            st.info("Performance report generation not implemented in this version.")
    
    with col3:
        if st.button("🔄 Reset Model"):
            if st.button("Confirm Reset", type="primary"):
                # Reinitialize model
                from feedback_nlp_model import FeedbackLoopNLPModel
                st.session_state.model = FeedbackLoopNLPModel(
                    model_name="distilbert-base-uncased",
                    num_labels=2,
                    learning_rate=2e-5,
                    feedback_weight=0.3
                )
                st.success("Model reset successfully!")
                st.rerun()

def settings_page():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("Model Configuration")
    
    # Model parameters
    model_name = st.selectbox(
        "Base Model:",
        ["distilbert-base-uncased", "bert-base-uncased"],
        index=0
    )
    
    learning_rate = st.slider(
        "Learning Rate:",
        min_value=1e-6,
        max_value=1e-3,
        value=2e-5,
        format="%.2e"
    )
    
    feedback_weight = st.slider(
        "Feedback Weight:",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    if st.button("Apply Settings"):
        st.info("Settings saved! Restart the app to apply changes.")
    
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Feedback History"):
            if hasattr(st.session_state.model, 'feedback_history'):
                st.session_state.model.feedback_history.clear()
                st.success("Feedback history cleared!")
    
    with col2:
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history.clear()
            st.success("Prediction history cleared!")
    
    st.subheader("Export Data")
    
    if st.button("📥 Export Feedback Data"):
        if hasattr(st.session_state.model, 'feedback_history') and st.session_state.model.feedback_history:
            feedback_data = []
            for feedback in st.session_state.model.feedback_history:
                feedback_data.append({
                    'text': feedback['text'],
                    'predicted_label': feedback['predicted_label'],
                    'actual_label': feedback['actual_label'],
                    'confidence': feedback['confidence'],
                    'user_rating': feedback['user_rating'],
                    'feedback_score': feedback['feedback_score'],
                    'timestamp': feedback['timestamp'].isoformat()
                })
            
            df = pd.DataFrame(feedback_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No feedback data to export.")

if __name__ == "__main__":
    main()
