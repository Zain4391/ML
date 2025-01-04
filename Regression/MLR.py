import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('student_scores_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# apply feature scaling
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

# split the data set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

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

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Predicted vs Actual Scores')
plt.legend()
plt.grid(True)
plt.show()