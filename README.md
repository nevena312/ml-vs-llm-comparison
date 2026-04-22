# ML vs LLM Comparison (using zero-shot and few-shot sentiment classification)

## Overview
This project compares traditional Machine Learning models with Large Language Models (LLMs).
The goal was to evaluate how well LLMs perform in zero-shot and few-shot settings compared to a supervised ML model trained on labeled data.

## Technologies used
- Python
- scikit-learn (Logistic Regression)
- transformers (BART-MNLI, FLAN-T5)
- numpy, pandas
- Google Colab

## Approach
- Preprocessed and split the dataset into training and test sets
- Trained a Logistic Regression model using labeled data
- Queried an LLM in:
  - zero-shot setting (no examples)
  - few-shot setting (with prompt examples)
- Compared predictions across approaches
- Evaluated performance using accuracy, precision, recall and F1-score

## Results
TF-IDF + Logistic Regression (40,000 samples)
→ Accuracy: ~0.90 | F1-score (positive): ~0.90
Zero-shot (BART-MNLI) (no task-specific training)
→ Accuracy: ~0.95 | F1-score (positive): ~0.95
Few-shot (FLAN-T5, 2+2 examples) (4 in-context examples)
→ Accuracy: ~0.89 | F1-score (positive): ~0.89
  
## Final conclusion
While LLMs offer remarkable flexibility and strong performance without task-specific training, traditional supervised models remain a robust and efficient solution when labeled data is available. The results emphasize the importance of selecting a model based on practical constraints rather than performance alone.

## Key Insights
- LLMs can outperform traditional models in zero-shot settings on certain datasets
- Few-shot performance depends heavily on prompt quality
- Traditional ML models remain more predictable and computationally efficient
