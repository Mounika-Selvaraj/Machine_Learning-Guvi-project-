# Used Car Price Prediction

This project uses machine learning techniques to predict the selling price of used cars based on various features such as mileage, engine capacity, maximum power, seats, vehicle age, and kilometers driven. The dataset used in this project contains various car attributes, and a Random Forest Regressor model is employed to predict the price of the car.

## Features

- **Mileage**: The car's fuel efficiency in km/l.
- **Engine**: The engine capacity in CC.
- **Max Power**: The maximum power of the car in BHP.
- **Seats**: The number of seats in the car.
- **Vehicle Age**: The number of years the car has been used.
- **Km Driven**: The total number of kilometers driven by the car.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab (for file uploads)

## Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd used-car-price-prediction
    ```

2. **Install dependencies**:
    Ensure you have Python 3.x installed. Then install the necessary packages:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3. **Upload Dataset**:
    - The dataset (`cardekho_dataset.csv`) should be uploaded through Google Colab when running the code, or you can manually place the dataset in the project directory.

4. **Run the Script**:
    - Open the `used_car_price_prediction.py` script in any IDE or run it directly in Jupyter Notebook/Google Colab.
    - The script will preprocess the data, train a model, and predict the price based on user input.

## Steps

1. **Data Preprocessing**:
    - Missing values in columns like mileage, engine capacity, max power, and seats are handled using median imputation.
    - The vehicle age is extracted from the `vehicle_age` column, which includes text and numeric values.
    - New features like mileage per year and kilometers driven per year are derived.

2. **Correlation Analysis**:
    - A correlation heatmap is displayed to show the relationships between different features and the target variable (selling price).
    - Features strongly correlated with the target are identified for model training.

3. **Model Training**:
    - The data is split into training and testing sets.
    - The data is standardized using `StandardScaler`.
    - A Random Forest Regressor model is trained on the scaled training data.

4. **Model Evaluation**:
    - The model's performance is evaluated using various metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

5. **Car Price Prediction**:
    - The script prompts the user for car details (mileage, engine capacity, max power, seats, vehicle age, and kilometers driven).
    - It calculates additional features like mileage per year and km per year.
    - The trained model predicts the selling price based on the entered details.

## Example Input/Output

### Input:
```
Enter the mileage (in km/l): 15.0
Enter the engine capacity (in CC): 1500
Enter the maximum power (in BHP): 100.0
Enter the number of seats: 5
Enter the vehicle age (in years): 5
Enter the total kilometers driven: 50000
```

### Output:
```
Car Details:
Mileage: 15.0 km/l
Engine Capacity: 1500 CC
Maximum Power: 100.0 BHP
Number of Seats: 5
Vehicle Age: 5 years
Kilometers Driven: 50000
Derived Mileage per Year: 3.00
Derived Kilometers per Year: 10000.00

Predicted Selling Price: ₹8,50,000.00
```

## Performance Metrics

The model's performance is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: The average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of the MSE, which gives an idea of the magnitude of errors.
- **R-squared (R2)**: A measure of how well the model explains the variance in the target variable.

## Conclusion

This project demonstrates how machine learning can be used to predict the selling price of used cars based on key features. By applying regression techniques, the model provides accurate price estimates based on user inputs. 

## Future Improvements

- Experimenting with different models like XGBoost or Gradient Boosting Regressor.
- Adding more features such as car brand and model-specific data.
- Implementing more advanced data preprocessing techniques to improve model accuracy.

## License

This project is licensed under the MIT License.

---

Let me know if you need any changes or additional sections!
