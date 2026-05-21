# Week 4 Progress Report

## Project Title
CNN-Based Plant Disease Classification Using the PlantVillage Dataset

---

## 1. Work Completed This Week

In Week 4, the project was finalized by improving model training stability, fixing dataset splitting issues, and preparing the system for deployment using a Streamlit web application.

The following tasks were completed:

- Fixed dataset splitting to ensure correct train/validation/test separation
- Eliminated potential data leakage issues from previous versions
- Finalized transfer learning pipeline using a pretrained ResNet model
- Applied full fine-tuning to improve model performance
- Trained the final model on the PlantVillage dataset
- Evaluated model using test dataset with classification metrics
- Generated confusion matrix and training loss visualization
- Built and tested a Streamlit application for real-time inference
- Exported final trained model for deployment

---

## 2. Final Model Improvements

In this week, the model was improved significantly compared to Week 3:

- Transitioned from partial training to full fine-tuning of the network
- Improved optimization using a lower learning rate (0.0001)
- Used learning rate scheduling to stabilize training
- Enhanced generalization through data augmentation
- Corrected dataset splitting strategy to ensure fair evaluation

The final model is based on a pretrained ResNet architecture, which was fully adapted to the PlantVillage dataset.

---

## 3. Evaluation Results

The model was evaluated on a held-out test set using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Key observations:

- High overall classification performance across most plant disease categories
- Some misclassifications occurred between visually similar diseases
- Healthy vs diseased classes were generally well distinguished
- Class imbalance still slightly affected rare categories

---

## 4. Error Analysis

A detailed error analysis was performed:

- Misclassified samples were visually inspected
- Most errors occurred in similar leaf disease patterns
- Small lesions and subtle visual differences were difficult for the model
- Some confusion exists between early-stage and late-stage diseases

This analysis helped identify limitations of the model and potential areas for improvement.

---

## 5. Streamlit Deployment

A Streamlit application was developed for real-time inference.

### Features:

- Image upload interface
- Real-time prediction using trained model
- Display of predicted plant disease class
- Support for multiple plant species and disease categories

### Workflow:

1. User uploads an image of a plant leaf
2. Image is preprocessed (resize, normalization)
3. Image is passed to the trained ResNet model
4. Model outputs predicted disease class
5. Result is displayed in the web interface

---

## 6. Challenges Encountered

Several challenges were addressed in Week 4:

- Fixing incorrect dataset splitting from previous weeks
- Ensuring consistent preprocessing between training and inference
- Managing model performance vs overfitting trade-off
- Aligning Streamlit class labels with training dataset classes
- Handling large number of output classes correctly

---

## 7. Final Project Outcome

By the end of Week 4, the project achieved:

- A fully trained deep learning model for plant disease classification
- A reproducible training pipeline
- A working Streamlit application for real-world usage
- Proper evaluation and error analysis pipeline

The system is now ready for final submission and demonstration.

---

## 8. Conclusion

Week 4 completed the full lifecycle of the project, including model optimization, evaluation, and deployment. The final system demonstrates strong performance in plant disease classification and provides a functional interface for real-time predictions using a trained deep learning model.
