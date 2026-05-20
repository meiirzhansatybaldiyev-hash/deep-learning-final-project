# Week 3 Progress Report

## Project Title
CNN-Based Plant Disease Classification Using the PlantVillage Dataset

---

## 1. Work Completed This Week

During Week 3, the focus was on improving the baseline model using transfer learning and implementing a more advanced evaluation pipeline. Compared to Week 2, this stage introduced a pretrained deep learning model, stronger data augmentation, and detailed error analysis.

The following tasks were completed:

- Integrated a pretrained ResNet-18 model for transfer learning
- Frozen convolutional backbone layers to preserve learned features
- Replaced the final fully connected layer for PlantVillage classification
- Applied advanced data augmentation techniques
- Improved preprocessing pipeline with separate training and testing transforms
- Trained the transfer learning model on the dataset
- Evaluated model performance on the test set
- Generated classification report and confusion matrix
- Implemented misclassification tracking for error analysis
- Visualized prediction examples for qualitative evaluation

---

## 2. Model Upgrade (Transfer Learning)

In this week, the baseline CNN was replaced with a pretrained deep learning model:

- Model: ResNet-18 (pretrained on ImageNet)
- Approach: Transfer learning
- Strategy: Freeze feature extractor layers and train only the classifier head

This allowed the model to leverage previously learned visual features such as edges, textures, and shapes, improving convergence speed and generalization.

---

## 3. Data Augmentation and Preprocessing

To improve generalization and reduce overfitting, the following augmentation techniques were applied during training:

- Random horizontal flipping
- Random rotation (±20 degrees)
- Color jitter (brightness and contrast adjustments)
- Image resizing to 224×224
- Normalization using ImageNet statistics

For validation and testing, only resizing and normalization were applied to ensure fair evaluation.

---

## 4. Training Setup

The training configuration was as follows:

| Parameter | Value |
|----------|------|
| Model | ResNet-18 (pretrained) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Epochs | 5 |
| Batch Size | 32 |
| Trainable Layers | Final fully connected layer only |

The model was trained on GPU when available.

---

## 5. Evaluation Results

The model was evaluated on the held-out test set.

### Metrics Used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The classification report showed improved performance compared to the baseline CNN from Week 2, especially in more visually distinct disease classes.

However, some confusion still exists between visually similar disease categories.

---

## 6. Confusion Matrix Analysis

A confusion matrix was generated to analyze classification performance across all classes.

### Key Observations:

- Most classes were classified correctly with high confidence
- Misclassifications mainly occurred between similar leaf diseases
- Certain plant species with visually similar symptoms were frequently confused
- Class imbalance may still influence performance for underrepresented categories

---

## 7. Error Analysis (Misclassified Samples)

A detailed error analysis was performed by storing and inspecting misclassified test samples.

### Findings:

- The model sometimes confuses diseases with similar visual patterns (e.g., leaf spots vs. early blight)
- Small lesions and subtle texture differences are difficult for the model to distinguish
- Some errors are likely caused by overlapping visual features across classes
- Augmentation improved robustness but did not fully eliminate misclassification cases

### Error Rate:
The final error rate was computed from the test predictions and reflects the proportion of incorrectly classified samples.

---

## 8. Qualitative Results (Example Predictions)

A set of misclassified images was visualized to better understand model behavior.

### Observations:

- Correct predictions are generally confident and visually consistent
- Incorrect predictions often involve subtle disease differences
- Some predictions show ambiguity even for human interpretation
- Model performs better on clear and distinct disease patterns

---

## 9. Key Improvements Over Week 2

Week 3 introduced significant improvements compared to the baseline model:

- Transfer learning instead of training from scratch
- Stronger feature extraction using ResNet-18
- Better generalization through data augmentation
- More robust evaluation using confusion matrix
- Introduction of error analysis and qualitative interpretation

Overall, the model shows improved stability and accuracy compared to Week 2 baseline CNN.

---

## 10. Challenges Encountered

Several challenges were observed during this week:

- Some disease classes remain visually very similar
- Class imbalance still affects minority classes
- Frozen layers limit full adaptation to dataset-specific features
- Misclassification analysis requires careful interpretation
- Training deeper models requires more computational resources

---

## 11. Plan for Week 4

The next week will focus on final improvements and project completion:

- Fine-tuning pretrained layers (unfreezing part of ResNet)
- Hyperparameter tuning (learning rate, batch size)
- Improving evaluation metrics visualization
- Final model optimization
- Full error analysis refinement
- Preparing final report and presentation slides
- Cleaning and organizing GitHub repository for submission

---

## 12. Conclusion

Week 3 significantly improved the project by introducing transfer learning and a more advanced evaluation pipeline. The model demonstrated better performance and generalization compared to the baseline CNN. Error analysis and confusion matrix evaluation provided deeper insight into model limitations, which will guide final improvements in Week 4.
