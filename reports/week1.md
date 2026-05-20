# Week 1 Progress Report

## Project Title
CNN-Based Plant Disease Classification Using the PlantVillage Dataset

---

## 1. Work Completed This Week

During Week 1, the project was initiated and the dataset was selected and explored. The main goal of this week was to understand the structure of the data and prepare the environment for model development in the upcoming weeks.

The following tasks were completed:

- Defined the project topic and scope
- Selected the PlantVillage dataset as the main dataset for the project
- Set up the project repository structure (folders for code, data, and reports)
- Installed required Python libraries
- Downloaded the dataset using KaggleHub
- Explored dataset directory structure
- Identified class labels (plant diseases and healthy leaves)
- Performed basic exploratory data analysis (EDA)
- Counted number of images per class
- Visualized sample images from different categories
- Generated dataset summary statistics

---

## 2. Dataset Description

The dataset used in this project is the PlantVillage dataset.

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/emmarex/plantdisease
- Type: Image dataset
- Format: JPG / PNG images
- Task: Multi-class image classification
- Approximate size: ~54,000 images
- Number of classes: 30+ plant disease categories

The dataset contains images of plant leaves from different crops such as tomato, potato, apple, and grape. Each image is labeled with either a disease type or a healthy class.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to better understand the dataset before training any models.

### Key Findings:

- The dataset includes multiple plant species and disease categories
- Each class corresponds to a specific disease or healthy condition
- There is a noticeable class imbalance between categories
- Some classes contain significantly more images than others
- Images are generally clear and high quality
- Backgrounds are relatively simple, which may help model performance

---

## 4. Class Distribution Analysis

The number of images per class was calculated to understand dataset balance.

Findings:

- Some plant disease categories are highly represented
- Some categories have fewer samples
- This imbalance may affect model performance if not addressed

Possible solutions planned for future weeks:

- Data augmentation
- Class weighting during training
- Balanced sampling techniques

---

## 5. Sample Image Visualization

Several sample images from different classes were displayed to verify data correctness.

Observations:

- Leaves show clear visual differences between healthy and diseased states
- Disease symptoms include spots, discoloration, and texture changes
- Image quality is sufficient for training deep learning models

---

## 6. Project Setup

A structured project repository was created to support development.

Recommended structure:

project-repo/
├── data/
├── notebooks/
├── src/
├── reports/
├── results/
├── requirements.txt
└── README.md

This structure ensures separation of concerns between data, code, and reports.

---

## 7. Tools and Technologies Used

The following tools and libraries were used in Week 1:

- Python
- KaggleHub (dataset download)
- Pandas (data analysis)
- Matplotlib (visualization)
- PIL (image processing)
- PyTorch (planned for model development in Week 2)

---

## 8. Challenges and Observations

No major technical problems were encountered during this week. However, several important observations were made:

- The dataset is large and requires efficient loading strategies
- Class imbalance exists and must be handled in later stages
- Dataset structure requires careful handling during loading
- Training deep learning models will require GPU support for efficiency

---

## 9. Plan for Week 2

The next week will focus on building and training the baseline model.

Planned tasks:

- Data preprocessing (resize, normalization, augmentation)
- Train/validation/test split (70/15/15)
- Implementation of baseline CNN model
- Definition of loss function and optimizer
- Training of initial model
- Evaluation of baseline performance
- Saving trained model checkpoints
- Documentation of initial results

---

## 10. Conclusion

Week 1 successfully established the foundation of the project. The dataset was selected, downloaded, and analyzed. Exploratory data analysis confirmed that the dataset is suitable for a deep learning-based image classification task.

The project is now ready for model development and training in Week 2.
