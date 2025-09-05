"""
Interactive Demo Interface for Feedback Loop NLP Model
This provides a user-friendly interface to interact with the model and provide feedback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os
import sys

# Add the current directory to path to import our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feedback_nlp_model import FeedbackLoopNLPModel

class InteractiveFeedbackDemo:
    """
    Interactive Streamlit demo for the feedback loop NLP model
    """
    
    def __init__(self):
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the model"""
        if 'model' not in st.session_state:
            with st.spinner('Loading NLP model...'):
                st.session_state.model = FeedbackLoopNLPModel(
                    model_name="distilbert-base-uncased",
                    num_labels=2,
                    learning_rate=2e-5,
                    feedback_weight=0.3
                )
                
                # Load some initial data and train
                train_texts, train_labels = st.session_state.model.load_open_source_data("imdb")
                train_dataset = st.session_state.model.preprocess_data(train_texts, train_labels)
                st.session_state.model.train_model(train_dataset)
        
        self.model = st.session_state.model
    
    def run(self):
        """Run the interactive demo"""
        st.set_page_config(
            page_title="Feedback Loop NLP Model Demo",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 Advanced Feedback Loop NLP Model")
        st.markdown("""
        This demo showcases an AI model that learns and improves through user feedback.
        The model uses a sophisticated feedback loop mechanism to continuously enhance its performance.
        """)
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Text Prediction", "Feedback Collection", "Performance Analytics", "Model Training"]
        )
        
        if page == "Text Prediction":
            self.text_prediction_page()
        elif page == "Feedback Collection":
            self.feedback_collection_page()
        elif page == "Performance Analytics":
            self.performance_analytics_page()
        elif page == "Model Training":
            self.model_training_page()
    
    def text_prediction_page(self):
        """Text prediction interface"""
        st.header("📝 Text Sentiment Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter your text:")
            user_text = st.text_area(
                "Type your text here:",
                placeholder="Enter a sentence or paragraph for sentiment analysis...",
                height=100
            )
            
            if st.button("🔍 Analyze Sentiment", type="primary"):
                if user_text.strip():
                    with st.spinner("Analyzing..."):
                        pred_label, confidence, details = self.model.predict_with_confidence(user_text)
                        
                        # Display results
                        sentiment = "Positive" if pred_label == 1 else "Negative"
                        confidence_pct = confidence * 100
                        
                        # Color coding
                        if pred_label == 1:
                            st.success(f"🎉 **Sentiment: {sentiment}** (Confidence: {confidence_pct:.1f}%)")
                        else:
                            st.error(f"😞 **Sentiment: {sentiment}** (Confidence: {confidence_pct:.1f}%)")
                        
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
                                color_discrete_map={'Negative': 'red', 'Positive': 'green'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Store prediction for feedback
                        st.session_state.last_prediction = {
                            'text': user_text,
                            'predicted_label': pred_label,
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        }
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.subheader("📈 Model Statistics")
            
            # Display model stats
            if hasattr(self.model, 'feedback_history') and self.model.feedback_history:
                total_feedback = len(self.model.feedback_history)
                avg_feedback_score = np.mean([f['feedback_score'] for f in self.model.feedback_history])
                
                st.metric("Total Feedback", total_feedback)
                st.metric("Avg Feedback Score", f"{avg_feedback_score:.2f}")
                
                # Recent feedback
                recent_feedback = list(self.model.feedback_history)[-5:]
                st.subheader("Recent Feedback")
                for i, feedback in enumerate(recent_feedback):
                    score = feedback['feedback_score']
                    color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                    st.write(f"{color} Score: {score:.2f}")
            else:
                st.info("No feedback collected yet.")
    
    def feedback_collection_page(self):
        """Feedback collection interface"""
        st.header("💬 Provide Feedback")
        
        if 'last_prediction' not in st.session_state:
            st.warning("Please make a prediction first on the Text Prediction page.")
            return
        
        last_pred = st.session_state.last_prediction
        
        st.subheader("Your Last Prediction:")
        st.write(f"**Text:** {last_pred['text']}")
        st.write(f"**Predicted Sentiment:** {'Positive' if last_pred['predicted_label'] == 1 else 'Negative'}")
        st.write(f"**Confidence:** {last_pred['confidence']:.3f}")
        
        st.subheader("Was the prediction correct?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Correct", type="primary"):
                actual_label = last_pred['predicted_label']
                user_rating = st.slider("Rate the prediction quality:", 1.0, 5.0, 4.0, 0.1)
                
                self.model.add_feedback(
                    last_pred['text'],
                    last_pred['predicted_label'],
                    actual_label,
                    last_pred['confidence'],
                    user_rating
                )
                
                st.success("✅ Feedback recorded! Thank you for helping improve the model.")
                del st.session_state.last_prediction
        
        with col2:
            if st.button("❌ Incorrect"):
                actual_label = 1 - last_pred['predicted_label']  # Flip the label
                user_rating = st.slider("Rate the prediction quality:", 1.0, 5.0, 2.0, 0.1)
                
                self.model.add_feedback(
                    last_pred['text'],
                    last_pred['predicted_label'],
                    actual_label,
                    last_pred['confidence'],
                    user_rating
                )
                
                st.error("❌ Feedback recorded! The model will learn from this mistake.")
                del st.session_state.last_prediction
        
        # Manual feedback section
        st.subheader("📝 Manual Feedback")
        
        manual_text = st.text_area("Enter text for manual feedback:")
        manual_sentiment = st.selectbox("Actual sentiment:", ["Positive", "Negative"])
        manual_rating = st.slider("Your rating:", 1.0, 5.0, 3.0, 0.1)
        
        if st.button("Submit Manual Feedback"):
            if manual_text.strip():
                actual_label = 1 if manual_sentiment == "Positive" else 0
                pred_label, confidence, _ = self.model.predict_with_confidence(manual_text)
                
                self.model.add_feedback(
                    manual_text,
                    pred_label,
                    actual_label,
                    confidence,
                    manual_rating
                )
                
                st.success("Manual feedback submitted successfully!")
            else:
                st.warning("Please enter text for feedback.")
    
    def performance_analytics_page(self):
        """Performance analytics and visualization"""
        st.header("📊 Performance Analytics")
        
        if not self.model.feedback_history:
            st.info("No feedback data available yet. Please provide some feedback first.")
            return
        
        # Convert feedback history to DataFrame
        feedback_data = []
        for feedback in self.model.feedback_history:
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
    
    def model_training_page(self):
        """Model training and retraining interface"""
        st.header("🏋️ Model Training")
        
        st.subheader("Current Model Status")
        
        if hasattr(self.model, 'feedback_history'):
            feedback_count = len(self.model.feedback_history)
            st.write(f"**Feedback samples collected:** {feedback_count}")
            
            if feedback_count >= 10:
                st.success("✅ Sufficient feedback for retraining!")
                
                if st.button("🔄 Retrain Model with Feedback", type="primary"):
                    with st.spinner("Retraining model with feedback data..."):
                        self.model.retrain_with_feedback(min_feedback_samples=10)
                        st.success("Model retrained successfully!")
                        
                        # Clear feedback history after retraining
                        self.model.feedback_history.clear()
                        st.info("Feedback history cleared after retraining.")
            else:
                st.warning(f"⚠️ Need at least 10 feedback samples for retraining. Current: {feedback_count}")
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Architecture:**")
            st.write(f"- Base Model: {self.model.model_name}")
            st.write(f"- Number of Labels: {self.model.num_labels}")
            st.write(f"- Learning Rate: {self.model.learning_rate}")
        
        with col2:
            st.write("**Feedback System:**")
            st.write(f"- Feedback Weight: {self.model.feedback_weight}")
            st.write(f"- Max Feedback History: {self.model.max_feedback_history}")
            st.write(f"- Device: {self.model.device}")
        
        st.subheader("Model Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Model"):
                self.model.save_model_state("saved_model.pth")
                st.success("Model saved successfully!")
        
        with col2:
            if st.button("📊 Generate Performance Report"):
                if hasattr(self.model, 'performance_history') and self.model.performance_history:
                    self.model.visualize_performance("performance_report.png")
                    st.success("Performance report generated!")
                else:
                    st.warning("No performance history available.")
        
        with col3:
            if st.button("🔄 Reset Model"):
                if st.button("Confirm Reset", type="primary"):
                    # Reinitialize model
                    st.session_state.model = FeedbackLoopNLPModel(
                        model_name="distilbert-base-uncased",
                        num_labels=2,
                        learning_rate=2e-5,
                        feedback_weight=0.3
                    )
                    st.success("Model reset successfully!")

def main():
    """Main function to run the interactive demo"""
    demo = InteractiveFeedbackDemo()
    demo.run()

if __name__ == "__main__":
    main()
