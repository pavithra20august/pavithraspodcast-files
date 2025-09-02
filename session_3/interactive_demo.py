#!/usr/bin/env python3
"""
Interactive Demo for Feedback Loop AI Model
This script allows you to interact with the AI model and provide real-time feedback.
"""

from feedback_ai_model import FeedbackAIModel
import numpy as np

def generate_sample_input():
    """Generate a random sample input for demonstration."""
    return np.random.rand(10).tolist()

def get_user_feedback():
    """Get feedback from the user."""
    while True:
        try:
            feedback = float(input("Rate the prediction (-1 to 1, where 1=excellent, -1=very poor): "))
            if -1 <= feedback <= 1:
                return feedback
            else:
                print("Please enter a value between -1 and 1.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    print("🤖 Interactive Feedback Loop AI Model Demo")
    print("=" * 50)
    print("This demo allows you to interact with an AI model that learns from your feedback.")
    print("The model will make predictions, and you can rate how good they are.")
    print("Over time, the model will learn to make better predictions based on your feedback.")
    print()
    
    # Initialize the model
    model = FeedbackAIModel(learning_rate=0.1)
    
    # Try to load existing model
    try:
        model.load_model()
        print("✅ Loaded existing model with training history.")
    except:
        print("🆕 Starting with a fresh model.")
    
    print("\nLet's begin! The model will generate predictions, and you'll provide feedback.")
    print("Type 'quit' to exit, 'stats' to see performance, or 'save' to save the model.\n")
    
    interaction_count = 0
    
    while True:
        interaction_count += 1
        print(f"\n--- Interaction {interaction_count} ---")
        
        # Generate input and prediction
        input_data = generate_sample_input()
        prediction, confidence = model.predict(input_data)
        
        print(f"Input features: {[f'{x:.2f}' for x in input_data[:5]]}... (showing first 5)")
        print(f"Model prediction: {prediction:.3f}")
        print(f"Confidence: {confidence:.3f}")
        
        # Get user command or feedback
        user_input = input("\nEnter feedback (-1 to 1), 'quit', 'stats', or 'save': ").strip().lower()
        
        if user_input == 'quit':
            break
        elif user_input == 'stats':
            stats = model.get_performance_stats()
            print("\n📊 Current Performance Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            continue
        elif user_input == 'save':
            model.save_model()
            print("💾 Model saved successfully!")
            continue
        else:
            try:
                feedback_score = float(user_input)
                if not (-1 <= feedback_score <= 1):
                    print("❌ Feedback must be between -1 and 1. Skipping this interaction.")
                    interaction_count -= 1
                    continue
            except ValueError:
                print("❌ Invalid input. Please enter a number between -1 and 1, or a command.")
                interaction_count -= 1
                continue
        
        # Provide feedback to the model
        feedback_type = "positive" if feedback_score > 0 else "negative" if feedback_score < 0 else "neutral"
        model.receive_feedback(input_data, prediction, feedback_score, feedback_type)
        
        # Show immediate performance update
        stats = model.get_performance_stats()
        print(f"✅ Feedback recorded! Average performance: {stats['average_score']:.3f}")
    
    # Final statistics
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n📊 Final Performance Statistics:")
    final_stats = model.get_performance_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Save the model
    save_choice = input("\n💾 Save the trained model? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        model.save_model()
        print("✅ Model saved to 'feedback_model.json'")
    
    print("\nThank you for helping train the AI model! 🤖✨")

if __name__ == "__main__":
    main()


