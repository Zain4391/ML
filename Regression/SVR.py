import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset
dataset = pd.read_csv('house_prices.csv')  # Replace with your dataset
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target variable

# Reshape target variable
y = y.reshape(len(y), 1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVR Model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')  # RBF Kernel
regressor.fit(X_train, y_train)

# Predict Test Results
y_pred = regressor.predict(X_test)

y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))
y_test = sc_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Model
from sklearn.metrics import r2_score, mean_absolute_error
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))


plt.scatter(sc_X.inverse_transform(X)[:, 0], sc_y.inverse_transform(y), color='red', label='Actual')  # Actual values
plt.scatter(sc_X.inverse_transform(X)[:, 0], sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue', label='Predicted')  # Predicted values
plt.title('SVR - Feature (Size) vs Price')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Actual vs Predicted Prices
y_pred = sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1))
y_actual = sc_y.inverse_transform(y)

plt.scatter(y_actual, y_pred, color='green')
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle='--')  # Perfect fit line
plt.title('SVR - Actual vs Predicted Prices')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.show()
