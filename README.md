# Sentiment Classification on Sentiment_140 Dataset

A deep learning project for binary sentiment classification (positive/negative) on Twitter data using a fine-tuned RoBERTa-Large transformer model.

## Dataset

- **Sentiment_140 Dataset**: 1.6 million tweets
- **Classes**: Binary (0: Negative, 1: Positive)
- **Split**: 80% train, 10% validation, 10% test

## Model Architecture

- **Base Model**: RoBERTa-Large (360M parameters)
- **Custom Components**:
  - Multi-pooling strategy (CLS, mean, max, attention pooling)
  - Multi-Sample Dropout for regularization
  - Attention-based pooling layer
  
## Training Configuration

- **Optimizer**: AdamW with Layer-wise Learning Rate Decay (LLRD factor: 0.9)
- **Learning Rate**: 1e-5 with cosine schedule and warmup
- **Batch Size**: 128 (effective batch: 256 with gradient accumulation)
- **Epochs**: 8
- **Mixed Precision**: BFloat16
- **Regularization**: Dropout (0.1), Weight Decay (0.01)

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 88.1% |
| Precision | 88.2% |
| Recall | 88.1% |
| F1 Score | 88.1% |

### Training Progress
- Best validation F1: **88.09%** achieved at Epoch 7
- Training time: ~5 hours on RTX 5090

## Key Features

1. **Text Preprocessing**: URL removal, mention handling, emoticon processing, repeated character normalization
2. **Data Augmentation**: Random word deletion during training
3. **Advanced Optimization**: LLRD, gradient clipping, warmup scheduling
4. **Hardware Optimization**: TF32, cuDNN benchmarking, mixed precision training

## Usage

```python
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("outputs/final_model")
model = torch.load("outputs/final_model/pytorch_model.bin")

# Predict sentiment
text = "I love this product!"
inputs = tokenizer(text, return_tensors="pt", max_length=96, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs['logits'], dim=1)
```

## Files

- `sentiment_classification.ipynb`: Complete training pipeline and error analysis

## Error Analysis

The model shows challenges with:
- **Sarcasm**: Literal interpretation of sarcastic statements
- **Negation**: Missing negation cues in complex sentences
- **Slang/Emojis**: Out-of-vocabulary tokens
- **Multi-topic tweets**: Conflicting sentiments in one tweet
- **Domain drift**: Specialized terminology

## Requirements

- PyTorch 2.7+
- Transformers (Hugging Face)
- pandas, numpy, scikit-learn
- tqdm

## Acknowledgments

- Sentiment_140 dataset from Stanford University
- RoBERTa model from Facebook AI
