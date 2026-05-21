# Final Project Report  
## CNN-Based Plant Disease Classification Using PlantVillage Dataset

---

# 1. Project Overview

This project focuses on building a deep learning system for automatic plant disease classification using the PlantVillage dataset. The system takes an image of a plant leaf as input and predicts the disease category or healthy status.

The project implements and compares multiple deep learning approaches:
- Baseline CNN model
- Transfer learning with ResNet
- Full fine-tuning of pretrained models
- Interactive Streamlit dashboard for model comparison and inference

The final result is a complete end-to-end machine learning system that includes training, evaluation, visualization, and deployment.

---

# 2. Dataset Description

**Dataset Name:** PlantVillage Dataset  
**Source:** Kaggle (emmarex/plantdisease)

### Dataset details:
- ~54,000 images
- 30+ plant disease classes
- RGB leaf images
- Structured in class-based folders

Each image represents a specific plant disease or healthy plant condition. The dataset is widely used for agricultural image classification tasks.

---

# 3. Project Objectives

The main objectives of this project are:

- Build a baseline CNN model for classification
- Improve performance using transfer learning
- Apply full fine-tuning for better adaptation
- Compare multiple models objectively
- Analyze errors and limitations
- Deploy a working Streamlit application

---

# 4. Methodology

## 4.1 Models Used

Three models were developed:

### Week 2 — Baseline CNN
A simple convolutional neural network:
- 3 convolutional layers
- max pooling layers
- fully connected classifier
- trained from scratch

### Week 3 — ResNet Feature Extraction
A pretrained :contentReference[oaicite:0]{index=0} model:
- pretrained on ImageNet
- frozen feature extractor
- trained classifier head only

### Week 4 — Full Fine-Tuning (Final Model)
- full ResNet fine-tuning
- all layers trainable
- lower learning rate (0.0001)
- learning rate scheduling

---

## 4.2 Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Batch Size: 32
- Image Size: 224 × 224
- Train/Validation/Test split: stratified

---

## 4.3 Data Preprocessing

- resizing images
- normalization using ImageNet statistics
- augmentation:
  - random horizontal flip
  - random rotation
  - color jitter

---

# 5. Weekly Progress Summary

---

## Week 1 — Dataset Preparation & Setup

### Completed:
- selected PlantVillage dataset
- defined problem statement
- created GitHub repository structure
- performed exploratory data analysis (EDA)
- analyzed dataset distribution

### Output:
- project proposal
- dataset overview
- initial repository setup

---

## Week 2 — Baseline CNN Model

### Completed:
- implemented CNN model from scratch
- built training pipeline
- trained baseline model
- evaluated initial performance

### Output:
- baseline accuracy results
- confusion matrix
- Week 2 report

### Observations:
- model learns basic features
- struggles with similar disease classes
- signs of overfitting

---

## Week 3 — Transfer Learning

### Completed:
- implemented ResNet18 transfer learning
- used pretrained ImageNet weights
- froze backbone layers
- trained classifier head
- performed error analysis

### Output:
- improved accuracy
- classification report
- confusion matrix
- misclassified sample analysis

### Key Insight:
Transfer learning significantly improved model performance and stability.

---

## Week 4 — Final Model & Deployment

### Completed:
- full fine-tuning of ResNet model
- fixed dataset splitting (no data leakage)
- improved training stability
- generated final evaluation metrics
- built Streamlit dashboard

### Output:
- final trained model file
- model comparison system
- confusion matrix analysis
- training curves
- Streamlit application

---

# 6. Model Comparison

Three models were evaluated:

| Model | Description |
|------|-------------|
| Week 2 | Baseline CNN |
| Week 3 | ResNet feature extraction |
| Week 4 | Full fine-tuned ResNet |

### Result Summary:
- Week 2: lowest performance, basic learning
- Week 3: strong improvement with pretrained features
- Week 4: best overall performance after fine-tuning

---

# 7. Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-class metrics

The model performs well overall, but confusion remains between visually similar disease classes.

---

# 8. Streamlit Application

A full interactive dashboard was developed using Streamlit.

## Features:

### 🔬 Prediction Module
- upload plant leaf image
- prediction from all models
- top-5 probability visualization

### 📊 Model Overview
- accuracy comparison
- precision, recall, F1 comparison

### 📉 Training Curves
- loss visualization per epoch

### 🔍 Per-Class Analysis
- class-wise precision, recall, F1
- filtering options

### 🔲 Confusion Matrix
- heatmap visualization
- normalized and raw views

---

## Workflow:

1. User uploads image
2. Image is preprocessed
3. Input passed into CNN / ResNet models
4. Each model returns prediction
5. Results are displayed in dashboard

---

# 9. Error Analysis

Main error patterns:

- confusion between similar diseases
- difficulty distinguishing early-stage symptoms
- misclassification of visually similar leaves
- imbalance between classes

---

# 10. Limitations

- dataset is clean and controlled (not real-world field conditions)
- limited generalization to real agricultural environments
- class imbalance affects rare categories
- high accuracy may not reflect real-world robustness

---

# 11. Conclusion

This project successfully implemented a full deep learning pipeline for plant disease classification.

Achievements:
- baseline CNN implementation
- transfer learning with ResNet
- full fine-tuning optimization
- model comparison system
- error analysis and visualization
- deployed Streamlit dashboard

The final system demonstrates strong performance on the PlantVillage dataset and provides a practical foundation for agricultural AI applications.

---

# 12. Future Improvements

- use stronger architectures (EfficientNet, Vision Transformers)
- train on real-world field datasets
- improve class imbalance handling
- deploy model to cloud (HuggingFace / AWS)
- create mobile application version
