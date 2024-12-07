# Used Car Price Prediction Project Report

## Overview

This project aims to predict the selling price of used cars based on various features such as mileage, engine capacity, maximum power, number of seats, vehicle age, and kilometers driven. The project utilizes a machine learning approach, where we implement a Random Forest Regressor model to predict the price of a car based on its characteristics.

The project is executed using **Google Colab**, leveraging Python libraries such as Pandas, NumPy, Scikit-learn, Seaborn, and Matplotlib for data processing, visualization, and machine learning. The dataset used in this project is from the `cardekho_dataset.csv` file, which contains information about various used cars and their selling prices.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Evaluation Metrics](#evaluation-metrics)
7. [User Input and Prediction](#user-input-and-prediction)
8. [Conclusion](#conclusion)

## Introduction

The goal of this project is to develop a model that can predict the selling price of used cars based on several features. The model will help users get an estimate of the price of a used car before making a purchase. To achieve this, we use a Random Forest Regressor model and preprocess the data for better accuracy and performance.

## Dataset Overview

The dataset used for this project contains the following columns:

- `selling_price`: The target variable (selling price of the car)
- `mileage`: Mileage of the car (in km/l)
- `engine`: Engine capacity of the car (in CC)
- `max_power`: Maximum power of the car (in BHP)
- `seats`: Number of seats in the car
- `vehicle_age`: Age of the car (in years)
- `km_driven`: Total kilometers driven by the car
- `brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`: Categorical features indicating the brand, model, seller type, fuel type, and transmission type.

## Data Preprocessing

### Handling Missing Values
We handle missing values by filling them with the median of the respective columns. This is done for the following columns:
- `mileage`
- `engine`
- `max_power`
- `seats`

### Feature Extraction
- **Vehicle Age**: The `vehicle_age` column is extracted by converting the textual format (e.g., "5 years") into numerical values (e.g., 5).
- **Mileage per Year**: We create a new feature `mileage_per_year` by dividing the mileage by the vehicle age.
- **Kilometers per Year**: A new feature `km_per_year` is created by dividing the kilometers driven by the vehicle age.

### Outlier Removal
The dataset is filtered to remove outliers from the `selling_price` using the Interquartile Range (IQR) method.

### Encoding Categorical Variables
Categorical features such as `brand`, `model`, `seller_type`, `fuel_type`, and `transmission_type` are encoded using **Label Encoding**.

## Feature Engineering

After preprocessing the data, we perform correlation analysis to identify the features that are most strongly correlated with the target variable (`selling_price`). Features with strong correlations (above a threshold of 0.5) are selected for modeling.

Additionally, highly correlated features (with correlation greater than 0.9) are checked and removed if necessary.

## Modeling

The machine learning model used for predicting the selling price of the car is **Random Forest Regressor**. We split the dataset into training and testing sets (80% training and 20% testing) using **train_test_split** from Scikit-learn.

We standardize the features using **StandardScaler** to ensure the model performs optimally.

The model is trained on the scaled training data and then tested on the scaled testing data to evaluate its performance.

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R2)**

These metrics help assess how well the model predicts the car prices and how close the predictions are to the actual values.

## User Input and Prediction

After training the model, the user can input the details of a car (such as mileage, engine capacity, maximum power, etc.) through the console. The model will then predict the selling price of the car based on these inputs.

### Sample Input:
```
Enter the details of the car:
Enter the mileage (in km/l): 15.0
Enter the engine capacity (in CC): 1500
Enter the maximum power (in BHP): 100.0
Enter the number of seats: 5
Enter the vehicle age (in years): 5
Enter the total kilometers driven: 50000
```

The predicted selling price will be displayed in INR.

## Conclusion

This project provides a practical approach to predicting the selling price of used cars using machine learning. By applying Random Forest Regressor and performing extensive data preprocessing, we can provide a reasonably accurate estimate of the selling price based on the car's features. This tool can be used by potential buyers or sellers of used cars to assess fair pricing before making decisions.

## Requirements

- Google Colab for running the project
- Python 3.x
- Required libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn

## Running the Project

1. Upload the dataset (`cardekho_dataset.csv`) to Google Colab.
2. Run the provided Python script.
3. Enter the details of a car when prompted to get the predicted selling price.

