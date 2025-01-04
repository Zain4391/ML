import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('vehicles.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.4f}")

# Scatter plot for actual data
plt.scatter(X, y, color='red')

# Sorted data for smooth curve
X_actual_sorted = np.sort(X, axis=0)  # Sort original features
X_poly_sorted = poly_reg.transform(X_actual_sorted)  # Apply transformation
y_poly_pred = regressor.predict(X_poly_sorted)  # Get predictions

# Polynomial regression curve
plt.plot(X_actual_sorted, y_poly_pred, color='blue')

# Labels
plt.title('Vehicle Price Prediction')
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Price ($1000)')
plt.show()
