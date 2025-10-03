## 🎯 Overview

This application leverages a fine-tuned BERT model to classify user queries into two categories:
- **👥 People Search**: Queries looking for specific individuals (e.g., "Find John Smith", "Contact Sarah Johnson")
- **🔎 General Search**: Queries for information, documents, or general content (e.g., "Search for restaurants", "Find AI documents")

The model achieves **99.2% accuracy** on validation data with precision and recall scores above 98%.

## ✨ Features

- **Real-time Classification**: Instant intent prediction as you type
- **Adjustable Threshold**: Customize classification sensitivity via sidebar slider
- **Detailed Analysis**: View raw probabilities, confidence scores, and decision logic
- **Sample Queries**: Pre-loaded test cases for quick evaluation
- **Model Information**: Display loaded model details and parameters
- **Responsive Design**: Clean, intuitive interface built with Streamlit

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install streamlit torch transformers pandas numpy scikit-learn
   ```

3. **Ensure your trained model is available:**
   - Place your trained BERT model in `./bert-intent-classifier/` directory
   - Or update the model path in the code

### Running the Application

```bash
# Run the full BERT integration
streamlit run app.py

```

Open your browser to `http://localhost:8501` to access the application.

## 📁 Project Structure

```
├── bert_intent_app.py          # Main Streamlit application
├── training_dataset.csv        # Training Dataset
├── test_app.py                 # Minimal test application
├── testing_dataset.csv         # Testing Dataset
├── num_label1.ipynb            # BERT model training notebook
├── bert-intent-classifier/     # Trained model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── README.md                   # This file
```

## 🤖 Model Details

### Architecture
- **Base Model**: BERT (bert-base-uncased)
- **Task**: Binary sequence classification
- **Output Classes**: 2 (People vs Search)
- **Parameters**: ~109M total, ~14M trainable (frozen layers strategy)

### Training Configuration
- **Learning Rate**: 5e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Optimization**: AdamW with warmup
- **Loss Function**: BCEWithLogitsLoss
- **Classification Threshold**: 0.6

### Performance Metrics
- **Accuracy**: 99.22%
- **Precision**: 100%
- **Recall**: 98.62%
- **F1-Score**: 99.30%

## 🎮 Usage Guide

### Basic Classification
1. Enter your query in the text input field
2. Click "🚀 Classify" or simply type to get instant results
3. View the predicted intent with confidence score

### Advanced Features
- **Threshold Adjustment**: Use the sidebar slider to modify classification sensitivity
- **Sample Testing**: Click pre-loaded sample queries for quick testing
- **Detailed Analysis**: Expand the "Detailed Analysis" section to see:
  - Raw probability scores
  - Decision logic explanation
  - Model parameters used

### Sample Queries to Try
- "Find John Doe" → People Search
- "Search for best restaurants" → General Search
- "Look up Mary Smith contact" → People Search
- "Research about machine learning" → General Search

## 🔧 Customization

### Modifying the Threshold
The default classification threshold is 0.6. Queries with probability > 0.6 are classified as "People Search". You can:
- Adjust via the sidebar slider in real-time
- Modify the default in the code: `threshold=0.6`

### Adding New Sample Queries
Edit the `sample_queries` list in `app.py`:
```python
sample_queries = [
    ("Your new query", "👥 or 🔎"),
    # Add more samples here
]
```

## 🛠️ Development

### Model Training
The BERT model was trained using the notebook `num_label1.ipynb` with:
- Custom dataset with balanced classes (56.74% positive class)
- Layer freezing strategy (last 2 encoder layers + classifier trainable)
- Stratified train/validation split (80/20)
- Custom trainer with BCE loss function

### Extending the Application
- **Multi-class Classification**: Modify `num_labels` and add more intent categories
- **Batch Processing**: Add file upload feature for bulk classification
- **API Integration**: Convert to FastAPI for programmatic access
- **User Analytics**: Add usage tracking and statistics

## 📊 Technical Notes

### Model Loading Priority
The app attempts to load models in this order:
1. `./bert-intent-classifier` (primary trained model)
2. `./results/checkpoint-200` (checkpoint fallback)
3. `bert-base-uncased` (base model warning)

### Performance Optimization
- Model loading is cached using `@st.cache_resource`
- CPU-optimized inference (can be modified for GPU)
- Efficient tokenization with max_length=64

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Troubleshooting

### Common Issues

**Model Not Loading**
- Ensure the model directory exists: `./bert-intent-classifier/`
- Check that all model files are present (config.json, pytorch_model.bin, tokenizer files)
- Verify model was saved correctly after training

**Blank Streamlit Interface**
- Try different port: `streamlit run app.py --server.port 8502`
- Clear browser cache or try incognito mode

**Import Errors**
- Install all requirements
- Check Python version compatibility (3.7+)

### Support
For issues or questions, please create an issue in the repository or contact the development team.


**Built with ❤️ using BERT, Streamlit, and PyTorch**
'''