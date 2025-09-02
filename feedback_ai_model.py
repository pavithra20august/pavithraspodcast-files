import numpy as np
import random
from typing import List, Tuple, Dict, Any
import json
import os
from datetime import datetime

class FeedbackAIModel:
    """
    A simple AI model that learns from feedback to improve its predictions.
    This model implements a basic reinforcement learning approach where it
    adjusts its behavior based on positive/negative feedback.
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_size: int = 1000):
        """
        Initialize the feedback AI model.
        
        Args:
            learning_rate: How quickly the model adapts to feedback
            memory_size: Maximum number of experiences to remember
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.experiences = []
        self.performance_history = []
        self.model_state = {
            'weights': np.random.randn(10),  # Simple feature weights
            'bias': 0.0,
            'confidence': 0.5
        }
        
    def predict(self, input_data: List[float]) -> Tuple[float, float]:
        """
        Make a prediction based on input data.
        
        Args:
            input_data: List of numerical features
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Ensure input is the right size
        if len(input_data) != len(self.model_state['weights']):
            # Pad or truncate input to match weights
            if len(input_data) < len(self.model_state['weights']):
                input_data.extend([0.0] * (len(self.model_state['weights']) - len(input_data)))
            else:
                input_data = input_data[:len(self.model_state['weights'])]
         
        # Simple linear prediction
        prediction = np.dot(self.model_state['weights'], input_data) + self.model_state['bias']
        
        # Apply sigmoid to get a probability-like output
        prediction = 1 / (1 + np.exp(-prediction))
        
        return prediction, self.model_state['confidence']
    
    def receive_feedback(self, input_data: List[float], prediction: float, 
                        feedback_score: float, feedback_type: str = "general"):
        """
        Receive feedback and update the model.
        
        Args:
            input_data: The input that was used for prediction
            prediction: The prediction that was made
            feedback_score: Score between -1 (very bad) and 1 (very good)
            feedback_type: Type of feedback for categorization
        """
        # Store the experience
        experience = {
            'input': input_data,
            'prediction': prediction,
            'feedback_score': feedback_score,
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experiences.append(experience)
        
        # Keep memory size manageable
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
        
        # Update model based on feedback
        self._update_model(input_data, prediction, feedback_score)
        
        # Track performance
        self.performance_history.append(feedback_score)
        
    def _update_model(self, input_data: List[float], prediction: float, feedback_score: float):
        """
        Update model weights based on feedback.
        """
        # Normalize input data
        input_array = np.array(input_data[:len(self.model_state['weights'])])
        if len(input_array) < len(self.model_state['weights']):
            input_array = np.pad(input_array, (0, len(self.model_state['weights']) - len(input_array)))
        
        # Calculate error (difference between desired and actual)
        # For positive feedback, we want to reinforce the prediction
        # For negative feedback, we want to move away from the prediction
        target = prediction + feedback_score * 0.1  # Small adjustment based on feedback
        
        # Calculate gradient
        error = target - prediction
        gradient = error * input_array
        
        # Update weights
        self.model_state['weights'] += self.learning_rate * gradient
        self.model_state['bias'] += self.learning_rate * error
        
        # Update confidence based on recent performance
        recent_performance = np.mean(self.performance_history[-10:]) if self.performance_history else 0
        self.model_state['confidence'] = max(0.1, min(0.9, 0.5 + recent_performance * 0.2))
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        """
        if not self.performance_history:
            return {
                'total_experiences': 0,
                'average_score': 0,
                'recent_average': 0,
                'confidence': self.model_state['confidence']
            }
        
        return {
            'total_experiences': len(self.experiences),
            'average_score': np.mean(self.performance_history),
            'recent_average': np.mean(self.performance_history[-10:]),
            'confidence': self.model_state['confidence'],
            'improvement_trend': self._calculate_improvement_trend()
        }
    
    def _calculate_improvement_trend(self) -> float:
        """
        Calculate if performance is improving over time.
        """
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = np.mean(self.performance_history[-10:])
        earlier = np.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else 0
        
        return recent - earlier
    
    def save_model(self, filename: str = "feedback_model.json"):
        """
        Save the model state to a file.
        """
        model_data = {
            'model_state': {
                'weights': self.model_state['weights'].tolist(),
                'bias': self.model_state['bias'],
                'confidence': self.model_state['confidence']
            },
            'experiences': self.experiences[-100:],  # Save last 100 experiences
            'performance_history': self.performance_history,
            'learning_rate': self.learning_rate,
            'memory_size': self.memory_size
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filename: str = "feedback_model.json"):
        """
        Load the model state from a file.
        """
        if not os.path.exists(filename):
            print(f"Model file {filename} not found. Starting with fresh model.")
            return
        
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.model_state['weights'] = np.array(model_data['model_state']['weights'])
        self.model_state['bias'] = model_data['model_state']['bias']
        self.model_state['confidence'] = model_data['model_state']['confidence']
        self.experiences = model_data.get('experiences', [])
        self.performance_history = model_data.get('performance_history', [])
        self.learning_rate = model_data.get('learning_rate', 0.1)
        self.memory_size = model_data.get('memory_size', 1000)


def demo_feedback_loop():
    """
    Demonstrate the feedback loop AI model in action.
    """
    print("🤖 Feedback Loop AI Model Demo")
    print("=" * 40)
    
    # Initialize the model
    model = FeedbackAIModel(learning_rate=0.1)
    
    # Simulate some training scenarios
    scenarios = [
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.8, 0.9, "positive"),
        ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], 0.2, 0.8, "positive"),
        ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.5, -0.3, "negative"),
        ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.9, -0.7, "negative"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1, 0.6, "positive"),
    ]
    
    print("\n📚 Training the model with feedback...")
    for i, (input_data, prediction, feedback_score, feedback_type) in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"  Input: {input_data[:3]}... (showing first 3 values)")
        print(f"  Prediction: {prediction:.3f}")
        print(f"  Feedback: {feedback_score:.1f} ({feedback_type})")
        
        # Get model's prediction
        model_pred, confidence = model.predict(input_data)
        print(f"  Model's prediction: {model_pred:.3f} (confidence: {confidence:.3f})")
        
        # Provide feedback
        model.receive_feedback(input_data, model_pred, feedback_score, feedback_type)
        
        # Show stats
        stats = model.get_performance_stats()
        print(f"  Average performance: {stats['average_score']:.3f}")
    
    print("\n🎯 Testing the trained model...")
    
    # Test with new inputs
    test_inputs = [
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        prediction, confidence = model.predict(test_input)
        print(f"Test {i}: Input {test_input[:3]}... → Prediction: {prediction:.3f} (confidence: {confidence:.3f})")
    
    # Show final statistics
    print("\n📊 Final Model Statistics:")
    final_stats = model.get_performance_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Save the model
    model.save_model()
    print("\n💾 Model saved to 'feedback_model.json'")
    
    return model


if __name__ == "__main__":
    # Run the demo
    trained_model = demo_feedback_loop()
    
    print("\n" + "=" * 40)
    print("🎉 Demo completed! The model has learned from feedback.")
    print("You can now use this model for your own applications.")
    print("=" * 40)

