# Emotion Recognition System

A multilingual emotion recognition system that classifies text into emotional categories using various machine learning approaches including FastText and transformer-based models.

## Overview

This project implements emotion classification for text data with support for:
- English and Ukrainian languages
- Multiple model architectures (FastText, BERT-based models)
- Web interface for real-time emotion detection
- Model interpretability through SHAP analysis

## Features

- **Multi-model Support**: FastText for lightweight classification and transformer models for high accuracy
- **Multilingual**: Built-in translation pipeline for Ukrainian to English
- **Web Interface**: Interactive Gradio interface for testing the models
- **Model Analysis**: SHAP integration for understanding model predictions
- **Distributed Training**: Accelerate support for efficient model training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
emotion-recognition/
├── train/
│   ├── train_fasttext.ipynb       # FastText model training notebook
│   ├── run_text_classification_no_trainer.py  # HuggingFace training script
│   └── accelerate_train.sh        # Training launch script
├── shap_analysis.ipynb            # Model interpretability analysis
├── web-interface.ipynb            # Gradio web interface
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Models

### FastText Model
- Lightweight and fast inference
- Trained on Ukrainian emotion dataset
- ~2.9M parameters
- 73.5% test accuracy

### Transformer Models
- **Base Model**: `intfloat/multilingual-e5-base`
- **English Emotion Model**: `j-hartmann/emotion-english-distilroberta-base`
- Support for 6 emotion classes: joy, sadness, anger, fear, love, surprise

## Usage

### Training

#### FastText Model
Open and run `train/train_fasttext.ipynb` to train the FastText model on your emotion dataset.

#### Transformer Model
```bash
cd train
bash accelerate_train.sh
```

### Web Interface

Launch the Gradio interface for real-time emotion detection:

```python
# Run the web-interface.ipynb notebook
# Or execute the interface code directly
```

The interface supports:
- Text input in Ukrainian or English
- Real-time emotion prediction
- Confidence scores for predictions

### Model Interpretation

Use the SHAP analysis notebook to understand model predictions:

```python
# Run shap_analysis.ipynb
```

This provides visual explanations of which words contribute most to specific emotion predictions.

## Emotion Categories

The system classifies text into the following emotions:
- **Joy** (радість)
- **Sadness** (сум)
- **Anger** (гнів)
- **Fear** (страх)
- **Love** (любов)
- **Surprise** (здивування)
- **Neutral** (нейтральність) - for some models

## Dataset

The project uses the `dair-ai/emotion` dataset with additional Ukrainian translations. The dataset includes:
- Training: 16,000 samples
- Validation: 2,000 samples
- Test: 2,000 samples

## Performance

| Model | Test Accuracy | Parameters |
|-------|--------------|------------|
| FastText | 73.5% | 2.9M |
| Transformer (e5-base) | ~90%+ | 278M |

### Per-class Performance (FastText)
- Joy: 83.7%
- Sadness: 80.4%
- Anger: 63.6%
- Fear: 59.9%
- Love: 53.4%
- Surprise: 50.6%

## Requirements

Key dependencies:
- `transformers~=4.56.1`
- `torch`
- `gradio`
- `shap~=0.48.0`
- `accelerate`
- `datasets`
- `matplotlib`
- `notebook`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Hugging Face for the transformers library and model hub
- DAIR-AI for the emotion dataset
- FastText team for the efficient text classification library