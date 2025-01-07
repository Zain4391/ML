import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


dataset = pd.read_csv('D:\\ML\\Kaggle_DS\\test_energy_data.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values

encoder_cols = ["Building Type", "Day of Week"]

# perform encoding
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), encoder_cols)], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X[:5])

# train and tes splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(f"Mean error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 score: {r2_score(y_test, y_pred)}")
