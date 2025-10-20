"""
Simple Command-Line Test Interface
Usage: python3 simple_test.py "Your text here"
"""

import sys
import os
from feedback_nlp_model import FeedbackLoopNLPModel

def test_text(text):
    """
    Test a single text input
    """
    print(f"🤖 Testing: '{text}'")
    print("-" * 50)
    
    # Initialize and load model
    model = FeedbackLoopNLPModel()
    if os.path.exists("simple_demo_model.pth"):
        model.load_model_state("simple_demo_model.pth")
        print("✅ Loaded trained model")
    else:
        print("⚠️  Using untrained model")
    
    # Make prediction
    pred_label, confidence, details = model.predict_with_confidence(text)
    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence_pct = confidence * 100
    
    # Display results
    print(f"🎯 Result: {sentiment}")
    print(f"📊 Confidence: {confidence_pct:.1f}%")
    print(f"📈 Probabilities:")
    print(f"   Negative: {details['probabilities'][0]:.3f}")
    print(f"   Positive: {details['probabilities'][1]:.3f}")
    
    return pred_label, confidence, details

def main():
    """
    Main function
    """
    if len(sys.argv) < 2:
        print("Usage: python3 simple_test.py 'Your text here'")
        print("\nExamples:")
        print("  python3 simple_test.py 'I love this movie!'")
        print("  python3 simple_test.py 'This is terrible'")
        print("  python3 simple_test.py 'Great quality product'")
        return
    
    # Get text from command line arguments
    text = " ".join(sys.argv[1:])
    test_text(text)

if __name__ == "__main__":
    main()
