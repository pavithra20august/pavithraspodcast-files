"""
Feedback Loop NLP Model - Complete Demo
This script demonstrates the full functionality of the feedback loop system.
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
    Complete demo of the feedback loop NLP model
    """
    print("🤖 Feedback Loop NLP Model - Complete Demo")
    print("=" * 60)
    
    try:
        # Import our model
        from feedback_nlp_model import FeedbackLoopNLPModel
        
        print("✅ Successfully imported model components")
        
        # Initialize model
        print("🔧 Initializing model...")
        model = FeedbackLoopNLPModel(
            model_name="distilbert-base-uncased",
            num_labels=2,
            learning_rate=2e-5,
            feedback_weight=0.3
        )
        
        print("✅ Model initialized successfully")
        
        # Load and prepare data
        print("📊 Loading open source data...")
        train_texts, train_labels = model.load_open_source_data("imdb")
        
        # Use a smaller subset for demo
        train_texts = train_texts[:200]  # Small dataset for quick demo
        train_labels = train_labels[:200]
        
        print(f"✅ Loaded {len(train_texts)} training samples")
        
        # Split data
        split_idx = int(0.8 * len(train_texts))
        eval_texts = train_texts[split_idx:]
        eval_labels = train_labels[split_idx:]
        train_texts = train_texts[:split_idx]
        train_labels = train_labels[:split_idx]
        
        print(f"📈 Training samples: {len(train_texts)}")
        print(f"📊 Evaluation samples: {len(eval_texts)}")
        
        # Train initial model
        print("🏋️ Training initial model...")
        train_dataset = model.preprocess_data(train_texts, train_labels)
        eval_dataset = model.preprocess_data(eval_texts, eval_labels)
        model.train_model(train_dataset, eval_dataset)
        
        print("✅ Initial training completed")
        
        # Demo predictions
        print("\n🎯 Demonstrating predictions...")
        
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
        
        print("📝 Making predictions on demo texts:")
        print("-" * 50)
        
        for i, text in enumerate(demo_texts, 1):
            pred_label, confidence, _ = model.predict_with_confidence(text)
            sentiment = "Positive" if pred_label == 1 else "Negative"
            confidence_pct = confidence * 100
            
            print(f"{i:2d}. '{text}'")
            print(f"    → {sentiment} (Confidence: {confidence_pct:.1f}%)")
        
        # Demo feedback loop
        print("\n🔄 Demonstrating feedback loop...")
        
        # Test samples for feedback
        test_samples = [
            ("This movie is absolutely terrible and boring!", 0),
            ("Fantastic performance and amazing storyline!", 1),
            ("Boring and predictable plot with poor acting.", 0),
            ("Outstanding cinematography and brilliant direction.", 1),
            ("Waste of time and money, completely disappointing.", 0),
        ]
        
        print("📝 Collecting feedback on test samples...")
        
        for i, (text, true_label) in enumerate(test_samples):
            # Make prediction
            pred_label, confidence, _ = model.predict_with_confidence(text)
            
            # Simulate user feedback (rating from 1-5)
            if pred_label == true_label:
                user_rating = 4.5  # High rating for correct prediction
            else:
                user_rating = 2.0  # Low rating for incorrect prediction
            
            # Add feedback
            model.add_feedback(text, pred_label, true_label, confidence, user_rating)
            
            print(f"   Sample {i+1}: Predicted {pred_label}, Actual {true_label}, "
                   f"Confidence {confidence:.3f}, Rating {user_rating}")
        
        print(f"✅ Collected {len(test_samples)} feedback samples")
        
        # Retrain with feedback
        print("🔄 Retraining model with feedback...")
        model.retrain_with_feedback(min_feedback_samples=3)
        
        print("✅ Feedback-based retraining completed")
        
        # Test improved model
        print("\n🎯 Testing improved model...")
        
        test_texts = [
            "I love this new product, it's amazing!",
            "This service is terrible and unhelpful.",
            "Great quality and fast delivery.",
            "Poor customer service and slow response.",
            "Excellent value for money, highly recommended!"
        ]
        
        print("📝 Testing improved predictions:")
        print("-" * 50)
        
        for text in test_texts:
            pred_label, confidence, _ = model.predict_with_confidence(text)
            sentiment = "Positive" if pred_label == 1 else "Negative"
            confidence_pct = confidence * 100
            print(f"   '{text}' → {sentiment} (Confidence: {confidence_pct:.1f}%)")
        
        # Save model state
        print("\n💾 Saving model state...")
        model.save_model_state("demo_model.pth")
        
        print("✅ Model state saved successfully")
        
        # Demo summary
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("📋 DEMO SUMMARY:")
        print(f"   ✅ Model trained on {len(train_texts)} samples")
        print(f"   ✅ Made predictions on {len(demo_texts)} demo texts")
        print(f"   ✅ Collected {len(test_samples)} feedback samples")
        print(f"   ✅ Retrained model with feedback")
        print(f"   ✅ Model saved to: demo_model.pth")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Next steps:")
        print("   • Run: python3 quick_test.py (test with sample data)")
        print("   • Run: python3 simple_test.py 'Your text here' (test your input)")
        print("   • Run: streamlit run interactive_demo.py (web interface)")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)
