import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the training data from train.csv
train_data = pd.read_csv('train.csv')

# Drop rows with NaN values in any column
train_data = train_data.dropna()

X_train = train_data[['x']]
y_train = train_data['y']

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Load the test data from test.csv
test_data = pd.read_csv('test.csv')

# Drop rows with NaN values in any column
test_data = test_data.dropna()

X_test = test_data[['x']]
y_test = test_data['y']

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error and R-squared (Coefficient of Determination)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared (Coefficient of Determination): {r_squared:.2f}")

# Create a scatter plot of the actual y values vs. predicted y values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()
plt.show()
