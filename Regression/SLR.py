import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# get the data
dataset = pd.read_csv('house_prices.csv')

# features and dependent variable seperated
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate regression
regressor = LinearRegression()

# train the model
regressor.fit(X_train, y_train)

# make prediction
y_pred = regressor.predict(X_test)

# plot the training results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("House prices vs. Sq feet (Training Set)")
plt.xlabel("House size (in sq ft)")
plt.ylabel("House prices (in $)")
plt.show()

# plot the test results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("House prices vs. Sq feet (Test Set)")
plt.xlabel("House size (in sq ft)")
plt.ylabel("House prices (in $)")
plt.show()