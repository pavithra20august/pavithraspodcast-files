# 🤖 Feedback Loop NLP Model - Demo

A sophisticated AI model that learns and improves through user feedback, implementing a comprehensive feedback loop mechanism for continuous learning and performance enhancement.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Run Complete Demo
```bash
python3 demo.py
```

### 3. Test with Your Input
```bash
# Test with sample data
python3 quick_test.py

# Test your own text
python3 simple_test.py 'Your text here'

# Interactive testing
python3 simple_demo.py
```

### 4. Unified Web Interface (Recommended)
```bash
pip3 install -r streamlit_requirements.txt
streamlit run streamlit_app.py
```

### 5. Legacy Web Interface
```bash
streamlit run interactive_demo.py
```

## 📁 Project Structure

```
├── demo.py                    # Complete demo script
├── feedback_nlp_model.py      # Core model implementation
├── streamlit_app.py          # Unified Streamlit web interface (NEW)
├── interactive_demo.py        # Legacy Streamlit interface
├── quick_test.py             # Test with sample data
├── simple_demo.py            # Simple functionality demo
├── simple_test.py            # Command-line text testing
├── requirements.txt          # Basic dependencies
├── streamlit_requirements.txt # Web interface dependencies
├── README.md                 # This file
└── demo_model.pth           # Trained model (generated after demo)
```

## 🎯 Features

- **Advanced NLP Model**: Built on DistilBERT transformer
- **Feedback Loop System**: Continuous learning from user feedback
- **Open Source Data**: Automatically loads IMDB dataset
- **Multiple Testing Options**: Command-line, interactive, and web interface
- **Model Persistence**: Save/load complete model state

## 🔄 How It Works

1. **Initial Training**: Model trains on open source sentiment data
2. **Prediction**: Makes sentiment predictions with confidence scores
3. **Feedback Collection**: Users provide feedback on prediction accuracy
4. **Continuous Learning**: Model retrains using accumulated feedback
5. **Performance Improvement**: Model gets better over time

## 📊 Testing Options

### Quick Test with Sample Data
```bash
python3 quick_test.py
```
Tests 20 predefined examples and shows accuracy metrics.

### Test Your Own Text
```bash
python3 simple_test.py 'I love this product!'
python3 simple_test.py 'This is terrible'
```

### Interactive Demo
```bash
python3 simple_demo.py
```
Interactive interface for testing and feedback collection.

### Unified Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
Comprehensive web interface with all features:
- 🏠 Home dashboard
- 📝 Text prediction
- 🔄 Feedback collection
- 📊 Performance analytics
- 🏋️ Model training
- ⚙️ Settings

### Legacy Web Interface
```bash
streamlit run interactive_demo.py
```
Basic web interface with core functionality.

## 🎛️ Model Configuration

The model uses these default settings:
- **Base Model**: DistilBERT (lightweight transformer)
- **Task**: Binary sentiment analysis (Positive/Negative)
- **Learning Rate**: 2e-5
- **Feedback Weight**: 0.3
- **Device**: CPU (for compatibility)

## 📈 Performance

The model achieves:
- **Training**: ~3 minutes on CPU
- **Inference**: ~0.1 seconds per prediction
- **Accuracy**: Improves with feedback (starts ~50%, improves to 80%+)
- **Confidence**: Provides prediction confidence scores

## 🔧 Customization

You can modify the model by editing `feedback_nlp_model.py`:
- Change base model (e.g., to BERT, RoBERTa)
- Adjust learning rate and training parameters
- Modify feedback scoring algorithm
- Add new evaluation metrics

## 🚀 Usage Examples

### Basic Prediction
```python
from feedback_nlp_model import FeedbackLoopNLPModel

model = FeedbackLoopNLPModel()
pred_label, confidence, details = model.predict_with_confidence("I love this!")
sentiment = "Positive" if pred_label == 1 else "Negative"
print(f"Sentiment: {sentiment}, Confidence: {confidence:.3f}")
```

### Feedback Collection
```python
# Add feedback to improve the model
model.add_feedback(text, predicted_label, actual_label, confidence, user_rating)

# Retrain with collected feedback
model.retrain_with_feedback(min_feedback_samples=10)
```

### Model Persistence
```python
# Save model state
model.save_model_state("my_model.pth")

# Load model state
model.load_model_state("my_model.pth")
```

## 🎉 Demo Results

After running the complete demo, you'll see:
- ✅ Model trained on 160 samples
- ✅ Predictions on 10 demo texts
- ✅ Feedback collection from 5 samples
- ✅ Model retraining with feedback
- ✅ Improved performance on test texts
- ✅ Model saved for future use

## 🔮 Next Steps

1. **Run the demo**: `python3 demo.py`
2. **Test with your data**: `python3 quick_test.py`
3. **Try the web interface**: `streamlit run interactive_demo.py`
4. **Customize the model**: Edit `feedback_nlp_model.py`
5. **Collect more feedback**: Use the interactive tools

## 📝 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for advancing AI through continuous learning and feedback.**