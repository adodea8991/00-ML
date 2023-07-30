import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('combined.csv')

# Split the data into features (x) and target (y)
x = data['x'].values.reshape(-1, 1)
y = data['y'].values

# Perform 70-30 train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Plot the data points and the linear regression model
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Model (70-30 Split)')
plt.show()

# Calculate the residuals
residuals = y_test - y_pred

# Plot the residuals
plt.scatter(x_test, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('Residuals')
plt.title('Residual Plot (70-30 Split)')
plt.show()
