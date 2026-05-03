# CNN-Based Plant Disease Classification Using the PlantVillage Dataset

## 1. Project Title

**CNN-Based Plant Disease Classification Using the PlantVillage Dataset**

---

## 2. Problem Statement

Plant diseases significantly reduce agricultural productivity and can lead to economic losses for farmers. Early and accurate detection of plant diseases is essential for improving crop management and ensuring food security.

The goal of this project is to build a deep learning model that can automatically classify plant leaf images into different disease categories. Given an image of a plant leaf, the model will predict the corresponding disease class or identify it as healthy.

This problem is useful because it enables scalable and automated plant disease detection, which can assist farmers and agricultural experts in making faster and more informed decisions. The final model will output a predicted label representing the disease type for each input image.

---

## 3. Dataset

This project uses the PlantVillage Dataset.

* **Source:** https://www.kaggle.com/datasets/emmarex/plantdisease
* **Number of examples:** ~54,000 images
* **Input features:** RGB images of plant leaves
* **Target labels:** Disease class (e.g., *Tomato___Late_blight*, *Apple___Scab*, *Healthy*)
* **Number of classes:** 30+ categories (depending on dataset version)
* **Data format:** Image files (JPEG/PNG) organized into class-specific folders
* **Structure:** Each folder corresponds to one class label

The dataset contains labeled images of both healthy and diseased plant leaves across multiple crop species. It is commonly used for image classification tasks in agriculture.

* **License / Usage:** Publicly available dataset on Kaggle for educational and research purposes (check Kaggle page for exact license terms).

---

## 4. Planned Method

### Baseline Model

A simple Convolutional Neural Network (CNN) will be implemented as a baseline.

* Few convolutional layers (2–3)
* Max pooling layers
* Fully connected output layer
* Purpose: establish a reference performance level

### Deep Learning Model

A pretrained **ResNet (e.g., ResNet-18 or ResNet-34)** will be used with transfer learning.

* Replace final classification layer
* Fine-tune on PlantVillage dataset
* Expected to significantly outperform baseline

### Loss Function

* Cross-entropy loss (suitable for multi-class classification)

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### Data Split Plan

* Training set: 70%
* Validation set: 15%
* Test set: 15%

The test set will be strictly held out and not used during model selection to avoid data leakage.

---

## 5. Expected Challenges

One potential challenge is class imbalance, where some disease categories may have significantly more images than others, leading to biased predictions. Another issue could be overfitting, especially when training deep models on relatively similar images.

Additionally, some images may have low variability (uniform backgrounds), which can reduce model generalization to real-world conditions. Training deep learning models such as ResNet may also require substantial computational resources (GPU), which could limit experimentation speed.

Finally, distinguishing between visually similar diseases may be difficult and could affect classification accuracy.

---

## 6. Weekly Plan

| Week   | Planned Work                                                    | Expected Output                       |
| ------ | --------------------------------------------------------------- | ------------------------------------- |
| Week 1 | Dataset selection, repository setup, exploratory data analysis  | Proposal, README, dataset summary     |
| Week 2 | Data preprocessing, train/validation/test split, baseline model | Baseline results and Week 2 report    |
| Week 3 | Deep learning model training and experiments                    | Model results, plots, error analysis  |
| Week 4 | Improvements, final evaluation, final report and presentation   | Final code, final report, slides/demo |

---
