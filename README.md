# 🍂 Leaf Classification

Leaf Classification is a project aimed at classifying leaves using a neural network architecture. Leveraging the Leaf Classification dataset, this project entails data preprocessing, exploratory data analysis, and model training to accurately classify leaves.

## Dataset

The Leaf Classification dataset consists of features extracted from images of plant leaves, including margin, shape, and texture attributes. Each sample in the dataset corresponds to a specific leaf species, making it suitable for supervised learning tasks. The dataset is preprocessed to handle missing values, duplicates, and corrupted images, ensuring data integrity and quality.

## Methodology
**1. Data Preparation**
   - **Data Cleaning**: Missing values, duplicates, and corrupted images are removed to ensure data cleanliness.
   - **Exploratory Data Analysis**: Correlation analysis and visualization techniques are employed to gain insights into the dataset's characteristics.
   - **Image Dataset Exploration**: Images are loaded, resized, and cleaned to facilitate further processing.

**2. Model Development**
   - **Neural Network Architecture**: A convolutional neural network (CNN) architecture is designed to extract features from leaf images and classify them into different species.
   - **Hyperparameter Tuning**: Various hyperparameters such as batch size, dropout rates, optimizers, regularization techniques, and learning rates are explored to optimize model performance.
   - **Training and Evaluation**: The model is trained on the preprocessed dataset, and its performance is evaluated using training and testing datasets.

## Model Architecture
The neural network architecture comprises multiple convolutional layers followed by max-pooling and dropout layers to prevent overfitting. The final dense layer predicts the probability distribution of leaf species using softmax activation. Hyperparameters are fine-tuned to enhance model performance and achieve higher accuracy.


## Results
The trained model demonstrates promising results, achieving high accuracy in classifying leaf species. Training and testing accuracy curves are plotted to visualize the model's performance over epochs. Through meticulous hyperparameter tuning and experimentation, the model achieves optimal accuracy while avoiding overfitting.

## Author
- **Shehab Adel Ramadan Moharram**
