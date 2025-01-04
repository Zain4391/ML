import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('house_price_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# as we do not have any categorical data we won't do encoding

# split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# perform MLR
ml_reg = LinearRegression()
ml_reg.fit(X_train, y_train)

# get predictions 
y_pred = ml_reg.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.4f}")
