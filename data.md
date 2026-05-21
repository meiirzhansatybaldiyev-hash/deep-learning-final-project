Dataset Overview
This dataset is a widely used, structured version of the PlantVillage Benchmark hosted on Kaggle. It contains thousands of clean, high-resolution RGB images of plant leaves captured in controlled laboratory settings against uniform backgrounds. The primary objective of this data is to train computer vision systems to identify distinct crops and diagnose specific bacterial, fungal, or viral diseases based on leaf visual features.

Core Statistics & Characteristics
Total Image Count: ~54,300 RGB images.

Classes: 38 distinct categorical classes.

Crop Varieties Included: 14 different plant species (such as Tomato, Potato, Pepper, Apple, Grape, Corn/Maize, Strawberry, Cherry, Peach, Blueberry, Squash, Raspberry, and Soybeans).

Conditions Covered: Divided across healthy leaf control groups and various plant diseases (e.g., Late Blight, Early Blight, Black Rot, Leaf Rust, Powdery Mildew, Bacterial Spot, and Target Spot).

Data Directory Structure
The repository is systematically organized using a class-based folder taxonomy. This makes it easy to load directly into machine learning pipelines using frameworks like PyTorch or TensorFlow (ImageDataGenerator / image_dataset_from_directory).

Every folder follows a strict naming convention combining the plant name and the identified health status:

Plaintext
plantdisease/
└── PlantVillage/
    ├── Pepper__bell___Bacterial_spot/
    ├── Pepper__bell___healthy/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── Potato___healthy/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Leaf_Mold/
    ├── Tomato___healthy/
    └── [Remaining 30 classes...]
