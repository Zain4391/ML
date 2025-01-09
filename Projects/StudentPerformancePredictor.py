import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('D:\\ML\\Kaggle_DS\\Student_Performance.csv')

# Encode extracurricular activities
le = LabelEncoder()
encode_col = 'Extracurricular Activities'
dataset[encode_col] = le.fit_transform(dataset[encode_col])

# Split features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Multi Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lin = lr.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
print(f"R2 score for Linear Regression: {r2_lin}")

# Decision Tree Regression
dr = DecisionTreeRegressor()
dr.fit(X_train, y_train)
y_pred_dt = dr.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"R2 score for Decision Tree: {r2_dt}")

# Random Forest Regression
Rfr = RandomForestRegressor(n_estimators=100)
Rfr.fit(X_train, y_train)
y_pred_rfr = Rfr.predict(X_test)
r2_rfr = r2_score(y_test, y_pred_rfr)
print(f"R2 score for Random Forest with 100 trees: {r2_rfr}")

# Support Vector Regression
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
r2_svr = r2_score(y_test, y_pred_svr)
print(f"R2 score for Support Vector Regression: {r2_svr}")

# Plot Linear regression predictions
plt.figure(figsize=(14, 8))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_lin)), y_pred_lin, color='red', label='Linear Regression Predictions', alpha=0.7)
plt.title("Linear Regression Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.show()

# Plot Decision Tree predictions
plt.figure(figsize=(14, 8))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_dt)), y_pred_dt, color='green', label='Decision Tree Predictions', alpha=0.7)
plt.title("Decision Tree Regression Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.show()

# Plot Random Forest predictions
plt.figure(figsize=(14, 8))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_rfr)), y_pred_rfr, color='orange', label='Random Forest Predictions', alpha=0.7)
plt.title("Random Forest Regression Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.show()

# Plot Support Vector Regression predictions
plt.figure(figsize=(14, 8))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_svr)), y_pred_svr, color='purple', label='SVR Predictions', alpha=0.7)
plt.title("Support Vector Regression Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.show()

# Bar chart for R2 scores
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR']
r2_scores = [r2_lin, r2_dt, r2_rfr, r2_svr]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_scores, color=['red', 'green', 'orange', 'purple'])
plt.title("R² Scores of Regression Models")
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.ylim(0.9, 1.0)  
plt.show()
