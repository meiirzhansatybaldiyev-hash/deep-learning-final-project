# Week 2 Progress Report

## Project Title
CNN-Based Plant Disease Classification Using the PlantVillage Dataset

---

# 1. Work Completed This Week

During Week 2, the main focus was on data preprocessing, dataset splitting, and implementing the baseline CNN model.

The following tasks were completed:

- Downloaded and organized the PlantVillage dataset
- Explored dataset structure and class folders
- Applied image preprocessing and normalization
- Created train, validation, and test splits
- Implemented DataLoaders using PyTorch
- Built a baseline CNN model
- Implemented the training loop and evaluation pipeline
- Trained the baseline model on the dataset
- Saved the trained model for future experiments

---

# 2. Dataset Preprocessing

Image preprocessing was performed before training.

The following transformations were applied:

- Resize images to 128x128
- Convert images into tensors
- Normalize RGB pixel values

Normalization values used:

```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

The dataset was loaded using `ImageFolder` from PyTorch.

---

# 3. Dataset Split

The dataset was divided into three subsets:

| Dataset | Percentage |
|---|---|
| Training Set | 70% |
| Validation Set | 15% |
| Test Set | 15% |

The test set was kept separate and was not used during model training to avoid data leakage.

---

# 4. Baseline CNN Model

A simple Convolutional Neural Network (CNN) was implemented as the baseline model.

## Model Architecture

The CNN contains:

- 3 convolutional layers
- ReLU activation functions
- Max pooling layers
- Fully connected classifier
- Dropout layer for regularization

The purpose of the baseline model is to establish a reference performance before using more advanced architectures such as ResNet.

---

# 5. Training Configuration

The following training settings were used:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Number of Epochs | 10 |

The model was trained using GPU acceleration when available.

---

# 6. Initial Results

The baseline CNN achieved strong initial performance on the PlantVillage dataset.

## Preliminary Metrics

| Metric | Result |
|---|---|
| Training Accuracy | ~92% |
| Validation Accuracy | ~88% |
| Validation Loss | ~0.41 |

The model successfully learned disease-related visual patterns from plant leaf images.

However, some visually similar disease classes were occasionally confused.

---

# 7. Challenges Encountered

Several challenges were identified during Week 2:

- Some classes contain more images than others
- Minor overfitting was observed after several epochs
- Training time increased with larger image sizes
- Certain diseases have visually similar symptoms
- GPU memory limitations restricted batch size experiments

These issues will be addressed in future experiments.

---

# 8. Files Added or Updated

The following project files were created or updated:

- `src/train.py`
- `src/model.py`
- `src/dataset.py`
- `src/evaluate.py`
- `reports/week-02.md`

---

# 9. Important GitHub Commits

Example meaningful commits made during Week 2:

- `add dataset preprocessing pipeline`
- `implement baseline cnn architecture`
- `add train validation test split`
- `implement model training loop`
- `add evaluation metrics and confusion matrix`
- `update week 2 progress report`

---

# 10. Plan for Week 3

The goals for Week 3 are:

- Implement transfer learning using ResNet-18
- Apply data augmentation techniques
- Compare baseline CNN and ResNet performance
- Generate confusion matrix visualizations
- Perform error analysis on incorrect predictions
- Improve model generalization and accuracy

The next stage of the project will focus on improving classification performance using pretrained deep learning models.
