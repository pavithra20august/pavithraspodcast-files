#!/usr/bin/env python3
"""
Visualization script for the Feedback Loop AI Model
This script creates plots to visualize the model's learning progress.
"""

import matplotlib.pyplot as plt
import numpy as np
from feedback_ai_model import FeedbackAIModel
import json

def create_learning_visualization(model_file="feedback_model.json"):
    """
    Create visualizations of the model's learning progress.
    """
    try:
        # Load the model data
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        
        performance_history = model_data.get('performance_history', [])
        
        if not performance_history:
            print("No performance history found in the model file.")
            return
        
        # Create the visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Feedback Loop AI Model - Learning Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance over time
        ax1.plot(performance_history, 'b-', linewidth=2, alpha=0.7)
        ax1.set_title('Performance Over Time')
        ax1.set_xlabel('Interaction Number')
        ax1.set_ylabel('Feedback Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: Moving average
        if len(performance_history) >= 5:
            window_size = min(5, len(performance_history) // 2)
            moving_avg = np.convolve(performance_history, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(performance_history)), moving_avg, 'g-', linewidth=2)
            ax2.set_title(f'Moving Average (Window: {window_size})')
            ax2.set_xlabel('Interaction Number')
            ax2.set_ylabel('Average Performance')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance distribution
        ax3.hist(performance_history, bins=10, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Performance Distribution')
        ax3.set_xlabel('Feedback Score')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative performance
        cumulative_performance = np.cumsum(performance_history)
        ax4.plot(cumulative_performance, 'purple', linewidth=2)
        ax4.set_title('Cumulative Performance')
        ax4.set_xlabel('Interaction Number')
        ax4.set_ylabel('Cumulative Score')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n📊 Learning Progress Summary:")
        print(f"Total interactions: {len(performance_history)}")
        print(f"Average performance: {np.mean(performance_history):.3f}")
        print(f"Best performance: {np.max(performance_history):.3f}")
        print(f"Worst performance: {np.min(performance_history):.3f}")
        print(f"Performance standard deviation: {np.std(performance_history):.3f}")
        
        # Calculate improvement trend
        if len(performance_history) >= 10:
            first_half = np.mean(performance_history[:len(performance_history)//2])
            second_half = np.mean(performance_history[len(performance_history)//2:])
            improvement = second_half - first_half
            print(f"Improvement trend: {improvement:.3f} ({'Improving' if improvement > 0 else 'Declining' if improvement < 0 else 'Stable'})")
        
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found. Run the demo first to generate data.")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def create_comparison_visualization():
    """
    Create a comparison visualization showing different learning rates.
    """
    print("🔄 Creating comparison visualization with different learning rates...")
    
    learning_rates = [0.05, 0.1, 0.2]
    models = []
    
    # Train models with different learning rates
    for lr in learning_rates:
        model = FeedbackAIModel(learning_rate=lr)
        
        # Simulate training with the same scenarios
        scenarios = [
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.8, 0.9),
            ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], 0.2, 0.8),
            ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.5, -0.3),
            ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.9, -0.7),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1, 0.6),
        ]
        
        for input_data, prediction, feedback_score in scenarios:
            model_pred, _ = model.predict(input_data)
            model.receive_feedback(input_data, model_pred, feedback_score)
        
        models.append(model)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    for i, (model, lr) in enumerate(zip(models, learning_rates)):
        performance = model.performance_history
        plt.plot(performance, label=f'Learning Rate: {lr}', linewidth=2, alpha=0.8)
    
    plt.title('Performance Comparison: Different Learning Rates', fontsize=14, fontweight='bold')
    plt.xlabel('Interaction Number')
    plt.ylabel('Feedback Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparison visualization saved as 'learning_rate_comparison.png'")

def main():
    """
    Main function to run visualizations.
    """
    print("📈 Feedback Loop AI Model - Learning Visualization")
    print("=" * 50)
    
    # Check if matplotlib is available
    try:
        import matplotlib
        print("✅ Matplotlib is available. Creating visualizations...")
    except ImportError:
        print("❌ Matplotlib not found. Please install it with: pip install matplotlib")
        return
    
    # Create learning progress visualization
    print("\n1. Creating learning progress visualization...")
    create_learning_visualization()
    
    # Create comparison visualization
    print("\n2. Creating learning rate comparison...")
    create_comparison_visualization()
    
    print("\n🎉 Visualization complete!")
    print("Generated files:")
    print("  - learning_progress.png (if model data exists)")
    print("  - learning_rate_comparison.png")

if __name__ == "__main__":
    main()
