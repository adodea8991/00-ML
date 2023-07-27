import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from CSV
data = pd.read_csv('house_data_generated.csv')

# Split the data into features (house size) and target (house price)
X = data['House Size'].values.reshape(-1, 1)
y = data['House Price'].values

# Divide the data into 70% training and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize the linear regression model
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Model')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Linear Regression Model - House Price Prediction')
plt.legend()
plt.show()

# Calculate relevant statistics about the model's efficiency
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared (Coefficient of Determination): {r2:.2f}")
