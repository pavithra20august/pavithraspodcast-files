"""
Advanced Feedback Loop NLP Model with Continuous Learning
This model implements a sophisticated feedback mechanism for improving NLP performance over time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset as HFDataset, load_dataset
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackDataset(Dataset):
    """Custom dataset class for handling feedback data"""
    
    def __init__(self, texts: List[str], labels: List[int], feedback_scores: List[float] = None):
        self.texts = texts
        self.labels = labels
        self.feedback_scores = feedback_scores or [1.0] * len(texts)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],
            'feedback_score': self.feedback_scores[idx]
        }

class FeedbackLoopNLPModel:
    """
    Advanced NLP model with feedback loop mechanism for continuous learning
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_labels: int = 2,
                 learning_rate: float = 2e-5,
                 feedback_weight: float = 0.3,
                 max_feedback_history: int = 1000):
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.feedback_weight = feedback_weight
        self.max_feedback_history = max_feedback_history
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        # Feedback tracking
        self.feedback_history = deque(maxlen=max_feedback_history)
        self.performance_history = []
        self.model_versions = []
        
        # Training configuration - Force CPU to avoid MPS issues
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Model: {model_name}, Labels: {num_labels}")
    
    def load_open_source_data(self, dataset_name: str = "imdb") -> Tuple[List[str], List[int]]:
        """
        Load open source text data for training
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_name)
            
            # Extract texts and labels
            train_texts = dataset['train']['text'][:5000]  # Limit for demo
            train_labels = dataset['train']['label'][:5000]
            
            # Add some additional open source datasets for diversity
            if dataset_name == "imdb":
                # Add sentiment analysis from other sources
                additional_texts = [
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
                additional_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                
                train_texts.extend(additional_texts)
                train_labels.extend(additional_labels)
            
            logger.info(f"Loaded {len(train_texts)} training samples")
            return train_texts, train_labels
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic data as fallback"""
        logger.info("Generating synthetic data as fallback")
        
        positive_texts = [
            "This is amazing and wonderful!",
            "I love this product, it's fantastic!",
            "Excellent quality and great service.",
            "Outstanding performance and reliability.",
            "Perfect solution for my needs.",
            "Highly recommended and worth it.",
            "Brilliant design and functionality.",
            "Superb experience and satisfaction.",
            "Top-notch quality and value.",
            "Exceptional results and delivery."
        ]
        
        negative_texts = [
            "This is terrible and disappointing.",
            "I hate this product, it's awful!",
            "Poor quality and bad service.",
            "Underwhelming performance and unreliability.",
            "Waste of money and time.",
            "Not recommended at all.",
            "Bad design and functionality.",
            "Terrible experience and frustration.",
            "Low-quality and overpriced.",
            "Disappointing results and delivery."
        ]
        
        texts = positive_texts + negative_texts
        labels = [1] * len(positive_texts) + [0] * len(negative_texts)
        
        return texts, labels
    
    def preprocess_data(self, texts: List[str], labels: List[int]) -> HFDataset:
        """
        Preprocess text data for training
        """
        logger.info("Preprocessing data...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=True, 
                max_length=512
            )
        
        # Create dataset
        dataset_dict = {'text': texts, 'label': labels}
        dataset = HFDataset.from_dict(dataset_dict)
        
        # Tokenize
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset: HFDataset, eval_dataset: HFDataset = None):
        """
        Train the model with feedback-aware training
        """
        logger.info("Starting model training...")
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            greater_is_better=True,
            report_to="none",  # Disable wandb for demo
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model('./best_model')
        self.tokenizer.save_pretrained('./best_model')
        
        logger.info("Model training completed and saved")
    
    def add_feedback(self, text: str, predicted_label: int, actual_label: int, 
                    confidence: float, user_rating: float = None):
        """
        Add feedback to the model for continuous learning
        """
        feedback_entry = {
            'timestamp': datetime.now(),
            'text': text,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'confidence': confidence,
            'user_rating': user_rating,
            'error': abs(predicted_label - actual_label),
            'feedback_score': self._calculate_feedback_score(
                predicted_label, actual_label, confidence, user_rating
            )
        }
        
        self.feedback_history.append(feedback_entry)
        logger.info(f"Added feedback: Error={feedback_entry['error']}, Score={feedback_entry['feedback_score']:.3f}")
    
    def _calculate_feedback_score(self, predicted: int, actual: int, 
                                confidence: float, user_rating: float = None) -> float:
        """
        Calculate feedback score based on prediction accuracy and user input
        """
        # Base score from prediction accuracy
        accuracy_score = 1.0 if predicted == actual else 0.0
        
        # Confidence penalty/bonus
        confidence_factor = confidence if predicted == actual else (1.0 - confidence)
        
        # User rating factor (if provided)
        user_factor = user_rating if user_rating is not None else 0.5
        
        # Weighted combination
        final_score = (
            accuracy_score * 0.4 +
            confidence_factor * 0.3 +
            user_factor * 0.3
        )
        
        return final_score
    
    def predict_with_confidence(self, text: str) -> Tuple[int, float, Dict]:
        """
        Make prediction with confidence score and detailed output
        """
        # Ensure model is on CPU to avoid MPS issues
        self.model.to('cpu')
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move inputs to CPU explicitly
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence = torch.max(probabilities).item()
            predicted_label = torch.argmax(logits, dim=-1).item()
        
        return predicted_label, confidence, {
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'logits': logits.cpu().numpy()[0].tolist()
        }
    
    def retrain_with_feedback(self, min_feedback_samples: int = 100):
        """
        Retrain model using accumulated feedback
        """
        if len(self.feedback_history) < min_feedback_samples:
            logger.info(f"Not enough feedback samples ({len(self.feedback_history)}/{min_feedback_samples})")
            return
        
        logger.info(f"Retraining with {len(self.feedback_history)} feedback samples...")
        
        # Extract feedback data
        feedback_texts = [entry['text'] for entry in self.feedback_history]
        feedback_labels = [entry['actual_label'] for entry in self.feedback_history]
        feedback_weights = [entry['feedback_score'] for entry in self.feedback_history]
        
        # Create weighted dataset
        feedback_dataset = FeedbackDataset(feedback_texts, feedback_labels, feedback_weights)
        
        # Convert to HuggingFace format
        dataset_dict = {
            'text': feedback_texts,
            'label': feedback_labels
        }
        hf_feedback_dataset = HFDataset.from_dict(dataset_dict)
        
        # Preprocess
        tokenized_feedback = self.preprocess_data(feedback_texts, feedback_labels)
        
        # Fine-tune with feedback data
        training_args = TrainingArguments(
            output_dir='./feedback_results',
            num_train_epochs=2,
            per_device_train_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./feedback_logs',
            logging_steps=50,
            save_strategy="epoch",
            report_to="none",
            disable_tqdm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_feedback,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        # Save feedback-trained model
        trainer.save_model('./feedback_model')
        
        logger.info("Feedback-based retraining completed")
    
    def evaluate_performance(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """
        Evaluate model performance on test data
        """
        logger.info("Evaluating model performance...")
        
        predictions = []
        confidences = []
        
        for text in test_texts:
            pred, conf, _ = self.predict_with_confidence(text)
            predictions.append(pred)
            confidences.append(conf)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        avg_confidence = np.mean(confidences)
        
        performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_confidence': avg_confidence,
            'total_predictions': len(predictions)
        }
        
        self.performance_history.append(performance_metrics)
        
        logger.info(f"Performance - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Avg Confidence: {avg_confidence:.3f}")
        
        return performance_metrics
    
    def visualize_performance(self, save_path: str = "performance_analysis.png"):
        """
        Create visualizations of model performance and feedback
        """
        if not self.performance_history:
            logger.warning("No performance history available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance over time
        epochs = range(1, len(self.performance_history) + 1)
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        for i, metric in enumerate(metrics):
            values = [perf[metric] for perf in self.performance_history]
            axes[0, 0].plot(epochs, values, marker='o', label=metric)
        
        axes[0, 0].set_title('Model Performance Over Time')
        axes[0, 0].set_xlabel('Evaluation Epoch')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Feedback distribution
        if self.feedback_history:
            feedback_scores = [entry['feedback_score'] for entry in self.feedback_history]
            axes[0, 1].hist(feedback_scores, bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Feedback Score Distribution')
            axes[0, 1].set_xlabel('Feedback Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Confidence distribution
        if self.performance_history:
            confidences = [perf['average_confidence'] for perf in self.performance_history]
            axes[1, 0].plot(epochs, confidences, marker='s', color='green')
            axes[1, 0].set_title('Average Confidence Over Time')
            axes[1, 0].set_xlabel('Evaluation Epoch')
            axes[1, 0].set_ylabel('Average Confidence')
            axes[1, 0].grid(True)
        
        # Error analysis
        if self.feedback_history:
            errors = [entry['error'] for entry in self.feedback_history]
            error_counts = pd.Series(errors).value_counts()
            axes[1, 1].pie(error_counts.values, labels=['Correct', 'Incorrect'], 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Prediction Accuracy from Feedback')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance visualization saved to {save_path}")
    
    def save_model_state(self, filepath: str):
        """Save complete model state including feedback history"""
        state = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'feedback_history': list(self.feedback_history),
            'performance_history': self.performance_history,
            'model_config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'learning_rate': self.learning_rate,
                'feedback_weight': self.feedback_weight
            }
        }
        
        torch.save(state, filepath)
        logger.info(f"Model state saved to {filepath}")
    
    def load_model_state(self, filepath: str):
        """Load complete model state including feedback history"""
        state = torch.load(filepath, map_location='cpu', weights_only=False)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.model.to('cpu')  # Ensure model is on CPU
        self.tokenizer = state['tokenizer']
        self.feedback_history = deque(state['feedback_history'], maxlen=self.max_feedback_history)
        self.performance_history = state['performance_history']
        
        logger.info(f"Model state loaded from {filepath}")

def main():
    """
    Main function to demonstrate the feedback loop NLP model
    """
    logger.info("Starting Feedback Loop NLP Model Demo")
    
    # Initialize model
    model = FeedbackLoopNLPModel(
        model_name="distilbert-base-uncased",
        num_labels=2,
        learning_rate=2e-5,
        feedback_weight=0.3
    )
    
    # Load and preprocess data
    train_texts, train_labels = model.load_open_source_data("imdb")
    train_dataset = model.preprocess_data(train_texts, train_labels)
    
    # Split data for evaluation
    split_idx = int(0.8 * len(train_texts))
    eval_texts = train_texts[split_idx:]
    eval_labels = train_labels[split_idx:]
    train_texts = train_texts[:split_idx]
    train_labels = train_labels[:split_idx]
    
    train_dataset = model.preprocess_data(train_texts, train_labels)
    eval_dataset = model.preprocess_data(eval_texts, eval_labels)
    
    # Train initial model
    model.train_model(train_dataset, eval_dataset)
    
    # Evaluate initial performance
    initial_performance = model.evaluate_performance(eval_texts, eval_labels)
    
    # Simulate feedback loop
    logger.info("Simulating feedback loop...")
    
    test_samples = [
        ("This movie is absolutely terrible!", 0),
        ("Fantastic performance and great story!", 1),
        ("Boring and predictable plot.", 0),
        ("Outstanding cinematography and direction.", 1),
        ("Waste of time and money.", 0)
    ]
    
    for text, true_label in test_samples:
        pred_label, confidence, _ = model.predict_with_confidence(text)
        
        # Simulate user feedback (rating from 1-5)
        user_rating = np.random.uniform(3, 5) if pred_label == true_label else np.random.uniform(1, 3)
        
        model.add_feedback(text, pred_label, true_label, confidence, user_rating)
        
        logger.info(f"Text: {text[:50]}...")
        logger.info(f"Predicted: {pred_label}, Actual: {true_label}, Confidence: {confidence:.3f}")
    
    # Retrain with feedback
    model.retrain_with_feedback(min_feedback_samples=5)
    
    # Evaluate improved performance
    improved_performance = model.evaluate_performance(eval_texts, eval_labels)
    
    # Create visualizations
    model.visualize_performance("nlp_feedback_performance.png")
    
    # Save model state
    model.save_model_state("feedback_nlp_model.pth")
    
    logger.info("Demo completed successfully!")
    logger.info(f"Initial F1 Score: {initial_performance['f1_score']:.3f}")
    logger.info(f"Improved F1 Score: {improved_performance['f1_score']:.3f}")
    logger.info(f"Improvement: {improved_performance['f1_score'] - initial_performance['f1_score']:.3f}")

if __name__ == "__main__":
    main()
