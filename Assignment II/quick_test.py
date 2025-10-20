"""
Quick Test Script - Test the model with your own input or sample data
"""

import os
from feedback_nlp_model import FeedbackLoopNLPModel

def test_with_sample_data():
    """
    Test the model with predefined sample data
    """
    print("🧪 Testing with Sample Data")
    print("=" * 50)
    
    # Initialize and load model
    model = FeedbackLoopNLPModel()
    if os.path.exists("simple_demo_model.pth"):
        model.load_model_state("simple_demo_model.pth")
        print("✅ Loaded trained model")
    else:
        print("⚠️  Using untrained model")
    
    # Sample test data
    test_data = [
        # Positive examples
        ("I absolutely love this movie!", 1),
        ("This is the best product ever!", 1),
        ("Excellent service and quality.", 1),
        ("Amazing experience, highly recommended!", 1),
        ("Outstanding performance and delivery.", 1),
        ("Fantastic! Couldn't be happier.", 1),
        ("Perfect solution for my needs.", 1),
        ("Brilliant work, well done!", 1),
        
        # Negative examples
        ("This movie is terrible!", 0),
        ("Worst product I've ever bought.", 0),
        ("Poor quality and bad service.", 0),
        ("Completely disappointed with this.", 0),
        ("Waste of money and time.", 0),
        ("Terrible experience, avoid this.", 0),
        ("Not worth the price at all.", 0),
        ("Awful quality, very disappointed.", 0),
        
        # Neutral/ambiguous examples
        ("The movie was okay.", 1),  # Slightly positive
        ("It's not bad, but not great either.", 0),  # Slightly negative
        ("Average quality product.", 0),  # Neutral-negative
        ("It works as expected.", 1),  # Neutral-positive
    ]
    
    print(f"Testing {len(test_data)} sample texts...")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for i, (text, expected_label) in enumerate(test_data, 1):
        # Make prediction
        pred_label, confidence, _ = model.predict_with_confidence(text)
        sentiment = "Positive" if pred_label == 1 else "Negative"
        expected_sentiment = "Positive" if expected_label == 1 else "Negative"
        
        # Check if prediction is correct
        is_correct = pred_label == expected_label
        if is_correct:
            correct_predictions += 1
        
        # Display results
        status = "✅" if is_correct else "❌"
        print(f"{i:2d}. {status} '{text}'")
        print(f"    Predicted: {sentiment} (Confidence: {confidence:.1%})")
        print(f"    Expected:  {expected_sentiment}")
        print()
    
    # Calculate and display accuracy
    accuracy = correct_predictions / total_predictions
    print("=" * 50)
    print(f"📊 Results Summary:")
    print(f"   Total tests: {total_predictions}")
    print(f"   Correct: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Average confidence: {sum(confidence for _, confidence, _ in [model.predict_with_confidence(text) for text, _ in test_data]) / total_predictions:.1%}")

def test_with_your_text():
    """
    Test the model with your own text input
    """
    print("📝 Test with Your Own Text")
    print("=" * 50)
    
    # Initialize and load model
    model = FeedbackLoopNLPModel()
    if os.path.exists("simple_demo_model.pth"):
        model.load_model_state("simple_demo_model.pth")
        print("✅ Loaded trained model")
    else:
        print("⚠️  Using untrained model")
    
    # Your test texts
    your_texts = [
        "I love this new AI model!",
        "This is terrible and doesn't work.",
        "Great job on the implementation!",
        "Poor performance and slow response.",
        "Amazing results, exactly what I needed!",
        "Waste of time, completely useless.",
        "Excellent work, very impressed!",
        "Not good at all, very disappointed.",
    ]
    
    print("Testing with your custom texts:")
    print("-" * 50)
    
    for i, text in enumerate(your_texts, 1):
        pred_label, confidence, details = model.predict_with_confidence(text)
        sentiment = "Positive" if pred_label == 1 else "Negative"
        confidence_pct = confidence * 100
        
        print(f"{i}. '{text}'")
        print(f"   → {sentiment} (Confidence: {confidence_pct:.1f}%)")
        print(f"   → Probabilities: Negative={details['probabilities'][0]:.3f}, Positive={details['probabilities'][1]:.3f}")
        print()

def test_single_input():
    """
    Test a single input text
    """
    print("🎯 Single Input Test")
    print("=" * 50)
    
    # Initialize and load model
    model = FeedbackLoopNLPModel()
    if os.path.exists("simple_demo_model.pth"):
        model.load_model_state("simple_demo_model.pth")
        print("✅ Loaded trained model")
    else:
        print("⚠️  Using untrained model")
    
    # Test single input
    test_text = "This AI model is absolutely fantastic and works perfectly!"
    
    print(f"Testing: '{test_text}'")
    print("-" * 50)
    
    pred_label, confidence, details = model.predict_with_confidence(test_text)
    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence_pct = confidence * 100
    
    print(f"Prediction: {sentiment}")
    print(f"Confidence: {confidence_pct:.1f}%")
    print(f"Detailed probabilities:")
    print(f"  Negative: {details['probabilities'][0]:.3f}")
    print(f"  Positive: {details['probabilities'][1]:.3f}")
    print(f"Raw logits:")
    print(f"  Negative: {details['logits'][0]:.3f}")
    print(f"  Positive: {details['logits'][1]:.3f}")

def main():
    """
    Main function to run different test modes
    """
    print("🤖 Feedback Loop NLP Model - Quick Test")
    print("=" * 60)
    print("Choose a test mode:")
    print("1. Test with sample data (recommended)")
    print("2. Test with your custom texts")
    print("3. Test single input")
    print("4. Run all tests")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        test_with_sample_data()
    elif choice == "2":
        test_with_your_text()
    elif choice == "3":
        test_single_input()
    elif choice == "4":
        print("\n" + "="*60)
        test_with_sample_data()
        print("\n" + "="*60)
        test_with_your_text()
        print("\n" + "="*60)
        test_single_input()
    else:
        print("Invalid choice. Running sample data test...")
        test_with_sample_data()

if __name__ == "__main__":
    main()
