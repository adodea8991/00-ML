import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term to the input features
X = np.c_[np.ones(X.shape[0]), X]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_cost(theta, X, y, m, lambda_reg):
    z = X @ theta
    h = sigmoid(z)
    regularization_term = (lambda_reg / (2 * m)) * (theta[1:] @ theta[1:])  # Regularization term, excluding bias term
    cost = (-y @ np.log(np.clip(h, 1e-15, 1 - 1e-15)) - (1 - y) @ np.log(np.clip(1 - h, 1e-15, 1 - 1e-15))) / m + regularization_term
    return cost

def gradient_descent(X, y, theta, alpha, iterations, lambda_reg):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        z = X @ theta
        h = sigmoid(z)
        gradient = (X.T @ (h - y)) / m
        # Add regularization term to the gradient for all theta values except bias term
        gradient[1:] += (lambda_reg / m) * theta[1:]
        theta -= alpha * gradient
        cost = calculate_cost(theta, X, y, m, lambda_reg)
        cost_history.append(cost)
    return theta, cost_history

# Initialize parameters and hyperparameters
theta = np.zeros(X_train.shape[1])
alpha = 0.01
iterations = 1000
lambda_reg = 1

# Train the model
theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations, lambda_reg)

# Make predictions on the test set
y_pred = (sigmoid(X_test @ theta) >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Confusion Matrix:")
print(confusion_df)
