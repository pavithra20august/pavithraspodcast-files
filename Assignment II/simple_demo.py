"""
Simple Demo Script for Feedback Loop NLP Model
This script demonstrates the core functionality without full training.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Simple demo function showcasing the feedback loop NLP model
    """
    logger.info("🚀 Starting Simple Feedback Loop NLP Model Demo")
    
    try:
        # Import our model
        from feedback_nlp_model import FeedbackLoopNLPModel
        
        logger.info("✅ Successfully imported model components")
        
        # Initialize model
        logger.info("🔧 Initializing model...")
        model = FeedbackLoopNLPModel(
            model_name="distilbert-base-uncased",
            num_labels=2,
            learning_rate=2e-5,
            feedback_weight=0.3
        )
        
        logger.info("✅ Model initialized successfully")
        
        # Load some sample data
        logger.info("📊 Loading sample data...")
        train_texts, train_labels = model.load_open_source_data("imdb")
        
        # Use a very small subset for quick demo
        train_texts = train_texts[:50]  # Very small for quick demo
        train_labels = train_labels[:50]
        
        logger.info(f"✅ Loaded {len(train_texts)} sample texts")
        
        # Demo text predictions (without full training)
        logger.info("🎯 Demonstrating text predictions...")
        
        demo_texts = [
            "This movie is absolutely fantastic!",
            "Terrible acting and poor storyline.",
            "Great cinematography and direction.",
            "Boring and predictable plot.",
            "Outstanding performance by the cast.",
            "Waste of time and money.",
            "Brilliant script and execution.",
            "Disappointing and overrated.",
            "Masterpiece of modern cinema.",
            "Confusing and poorly edited."
        ]
        
        logger.info("📝 Making predictions on demo texts:")
        
        for i, text in enumerate(demo_texts):
            try:
                pred_label, confidence, _ = model.predict_with_confidence(text)
                sentiment = "Positive" if pred_label == 1 else "Negative"
                logger.info(f"   {i+1:2d}. '{text[:40]}...' -> {sentiment} (Confidence: {confidence:.3f})")
            except Exception as e:
                logger.warning(f"   {i+1:2d}. Error predicting '{text[:40]}...': {str(e)}")
        
        # Demo feedback collection
        logger.info("🔄 Demonstrating feedback collection...")
        
        # Simulate some feedback
        feedback_samples = [
            ("This movie is absolutely fantastic!", 1, 0.85, 4.5),  # text, true_label, confidence, user_rating
            ("Terrible acting and poor storyline.", 0, 0.92, 4.0),
            ("Great cinematography and direction.", 1, 0.78, 3.5),
            ("Boring and predictable plot.", 0, 0.88, 4.2),
            ("Outstanding performance by the cast.", 1, 0.95, 4.8)
        ]
        
        for text, true_label, confidence, user_rating in feedback_samples:
            # Make prediction
            pred_label, pred_confidence, _ = model.predict_with_confidence(text)
            
            # Add feedback
            model.add_feedback(text, pred_label, true_label, pred_confidence, user_rating)
            
            logger.info(f"   Added feedback: '{text[:30]}...' -> Pred: {pred_label}, Actual: {true_label}, Rating: {user_rating}")
        
        logger.info(f"✅ Collected {len(feedback_samples)} feedback samples")
        
        # Show feedback statistics
        if hasattr(model, 'feedback_history') and model.feedback_history:
            feedback_scores = [f['feedback_score'] for f in model.feedback_history]
            avg_score = sum(feedback_scores) / len(feedback_scores)
            logger.info(f"📊 Average feedback score: {avg_score:.3f}")
            logger.info(f"📊 Total feedback samples: {len(model.feedback_history)}")
        
        # Demo model persistence
        logger.info("💾 Demonstrating model persistence...")
        model.save_model_state("simple_demo_model.pth")
        logger.info("✅ Model state saved to simple_demo_model.pth")
        
        # Demo model loading
        logger.info("📂 Demonstrating model loading...")
        new_model = FeedbackLoopNLPModel()
        new_model.load_model_state("simple_demo_model.pth")
        logger.info("✅ Model state loaded successfully")
        
        # Verify loaded model works
        test_text = "This is a test sentence for verification."
        pred_label, confidence, _ = new_model.predict_with_confidence(test_text)
        sentiment = "Positive" if pred_label == 1 else "Negative"
        logger.info(f"✅ Loaded model prediction: '{test_text}' -> {sentiment} (Confidence: {confidence:.3f})")
        
        logger.info("🎉 Simple demo completed successfully!")
        
        # Summary
        logger.info("📋 DEMO SUMMARY:")
        logger.info(f"   ✅ Model initialized and ready")
        logger.info(f"   ✅ Made predictions on {len(demo_texts)} demo texts")
        logger.info(f"   ✅ Collected {len(feedback_samples)} feedback samples")
        logger.info(f"   ✅ Model persistence and loading verified")
        logger.info(f"   ✅ Model saved to: simple_demo_model.pth")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Simple demo completed successfully!")
        print("   - Model functionality verified")
        print("   - Feedback system working")
        print("   - Model persistence working")
        print("   - Model saved to: simple_demo_model.pth")
        print("\n🚀 To run the full interactive demo:")
        print("   streamlit run interactive_demo.py")
        print("\n🚀 To run the full training demo:")
        print("   python3 demo_script.py")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)
