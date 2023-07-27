import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Handling missing values by filling with mean
mean_y_train = np.mean(train_data['y'])
train_data['y'].fillna(mean_y_train, inplace=True)

mean_y_test = np.mean(test_data['y'])
test_data['y'].fillna(mean_y_test, inplace=True)

# Extract features (X) and target (y)
X_train = train_data[['x']].values
y_train = train_data['y'].values

X_test = test_data[['x']].values
y_test = test_data['y'].values

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Residual Analysis
plt.scatter(y_train_pred, train_residuals, color='blue', alpha=0.5, label='Training Data')
plt.scatter(y_test_pred, test_residuals, color='green', alpha=0.5, label='Test Data')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residuals')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Analysis')
plt.show()

# Output the relevant statistics
print(f"Mean Squared Error (Train): {train_mse:.2f}")
print(f"Mean Squared Error (Test): {test_mse:.2f}")
print(f"R-squared (Coefficient of Determination) - Train: {train_r2:.2f}")
print(f"R-squared (Coefficient of Determination) - Test: {test_r2:.2f}")
