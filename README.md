# Used Car Price Prediction - Project Report

## Project Overview

This project aims to predict the selling price of used cars based on various features such as mileage, engine capacity, maximum power, number of seats, vehicle age, and kilometers driven. The project uses a machine learning model, specifically a Random Forest Regressor, to make predictions based on historical data of used car listings. The project involves data cleaning, feature engineering, correlation analysis, model training, and evaluation.

## Tools and Libraries Used

The following libraries and tools are used in this project:

- **Google Colab**: Platform for running the code in a cloud environment.
- **Pandas**: Data manipulation and analysis library.
- **Numpy**: Library for numerical computations.
- **Matplotlib and Seaborn**: Visualization libraries used for plotting graphs.
- **Scikit-learn**: Machine learning library that provides tools for model training, evaluation, and preprocessing.
- **LabelEncoder**: For encoding categorical variables into numeric values.
- **RandomForestRegressor**: Machine learning model used for predicting the selling price of used cars.
- **StandardScaler**: For feature scaling before model training.

## Dataset

The dataset used in this project is the "cardekho_dataset.csv" file, which contains details about various used cars, including:

- `brand`: Brand of the car.
- `model`: Model of the car.
- `seller_type`: Type of seller (individual/dealer).
- `fuel_type`: Type of fuel used (petrol/diesel).
- `transmission_type`: Type of transmission (manual/automatic).
- `mileage`: Mileage of the car (in km/l).
- `engine`: Engine capacity (in CC).
- `max_power`: Maximum power (in BHP).
- `seats`: Number of seats in the car.
- `vehicle_age`: Age of the vehicle (in years).
- `km_driven`: Total kilometers driven by the car.
- `selling_price`: The target variable (selling price of the car).

## Data Preprocessing

The dataset undergoes several preprocessing steps:

1. **Handling Missing Values**: Missing values for columns like `mileage`, `engine`, `max_power`, and `seats` are filled with the median values of the respective columns.
   
2. **Feature Engineering**:
   - The `vehicle_age` column is converted into numeric values by extracting the year from the string representation.
   - New features like `mileage_per_year` and `km_per_year` are created to represent the car's yearly mileage and kilometers driven.

3. **Outlier Detection**: Outliers in the `selling_price` column are removed using the IQR (Interquartile Range) method.

4. **Label Encoding**: Categorical columns like `brand`, `model`, `seller_type`, `fuel_type`, and `transmission_type` are encoded into numeric values using `LabelEncoder`.

## Exploratory Data Analysis (EDA)

1. **Correlation Analysis**: A correlation matrix is plotted to visualize the relationships between different features and the target variable (`selling_price`). Strong correlations with the target variable are identified.
   
2. **Feature Selection**: Based on the correlation with the target variable, important features are selected for model training.

3. **Visualization**: A heatmap of the correlation matrix is displayed to understand how different features are related to each other and to the target variable.

## Model Development

1. **Train-Test Split**: The dataset is split into training and testing sets (80% train, 20% test).

2. **Feature Scaling**: The features are scaled using `StandardScaler` to ensure that all features are on the same scale before training the model.

3. **Random Forest Regressor**: A Random Forest Regressor model is trained using the scaled training data. The model is used to predict the selling price of used cars based on the features provided.

## Model Evaluation

The performance of the model is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in the predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **Root Mean Squared Error (RMSE)**: The square root of the MSE, providing a measure of the magnitude of error.
- **R-squared (R2)**: Measures how well the model explains the variance in the target variable.

## Prediction

The model allows users to enter details about a used car (mileage, engine capacity, maximum power, seats, vehicle age, and kilometers driven) and predicts the selling price of the car.

### Input:
The user provides the following information:
- Mileage (in km/l)
- Engine capacity (in CC)
- Maximum power (in BHP)
- Number of seats
- Vehicle age (in years)
- Kilometers driven

### Output:
The model predicts the selling price of the car and displays the entered details along with the predicted price.

## Example Output

```
Enter the details of the car:
Enter the mileage (in km/l): 15.0
Enter the engine capacity (in CC): 1500
Enter the maximum power (in BHP): 120.0
Enter the number of seats: 5
Enter the vehicle age (in years): 5
Enter the total kilometers driven: 50000

Car Details:
Mileage: 15.0 km/l
Engine Capacity: 1500 CC
Maximum Power: 120.0 BHP
Number of Seats: 5
Vehicle Age: 5 years
Kilometers Driven: 50000
Derived Mileage per Year: 3.00
Derived Kilometers per Year: 10000.00

Predicted Selling Price: ₹5,35,000.00
```

## Conclusion

This project provides a comprehensive solution for predicting the selling price of used cars based on various features. It demonstrates the application of machine learning techniques, including data preprocessing, feature engineering, and model evaluation, to build a predictive model. The model can be further improved with additional data and more advanced algorithms.

## Future Enhancements

- Implementing more advanced models like XGBoost or Gradient Boosting to improve prediction accuracy.
- Adding support for more features such as car color, previous accidents, etc.
- Developing a user-friendly web application to make predictions.

## How to Run the Code

1. Upload the dataset (`cardekho_dataset.csv`) to Google Colab.
2. Copy and paste the provided code into a Google Colab notebook.
3. Run the code sequentially to preprocess the data, train the model, and predict the selling price of a car.
