# Neural Network Training and Grid Search

## Overview
This project aims to apply machine learning techniques to a dataset (`XY_train.csv`) by performing the following tasks:

1. **Data Preprocessing**:
    - Handling missing data using **KNN Imputation** for columns `ANNUAL_MILEAGE` and `CREDIT_SCORE`.
    - Encoding categorical features like `GENDER`, `EDUCATION`, `INCOME`, `VEHICLE_YEAR`, and `VEHICLE_TYPE` into numerical values.
    - Normalizing features using **MinMaxScaler** to scale values between 0 and 1 for optimal performance in the neural network.

2. **Model Training**:
    - **MLPClassifier** (Multilayer Perceptron Classifier) is used for training the neural network model. 
    - The model is evaluated for performance on both training and testing data using **accuracy** and **F1-score**.

3. **Hyperparameter Tuning**:
    - **GridSearchCV** is used to find the best hyperparameters for the **MLPClassifier** model. The tuned hyperparameters include:
      - `hidden_layer_sizes`
      - `activation`
      - `alpha`
      - `learning_rate`
      - `learning_rate_init`
      - `max_iter`
      - `solver`
      - `early_stopping`
    - The tuning aims to improve the model's performance on the test set.

4. **Model Evaluation**:
    - **Accuracy** and **F1-score** are used to assess the model's classification performance.
    - A **Confusion Matrix** is generated to understand the number of true positives, true negatives, false positives, and false negatives.

5. **Visualization**:
    - **Heatmaps** are generated to visualize the effects of hyperparameters during **GridSearchCV**. These heatmaps display how different combinations of hyperparameters impact the model's performance.

## Requirements

To run this project, the following Python libraries are required:

- `pandas`: For data manipulation.
- `numpy`: For numerical computations.
- `matplotlib`: For plotting and visualizations.
- `seaborn`: For heatmaps and advanced visualizations.
- `scikit-learn`: For building and tuning machine learning models.
- `scipy`: For statistical functions used in KNN imputation.

You can install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
