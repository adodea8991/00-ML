import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data from emails.csv
data = pd.read_csv('emails.csv')

# Drop the 'Email No.' column and use the rest as features
X = data.drop(columns=['Email No.', 'Prediction'])

# Use the 'Prediction' column as the target variable
y = data['Prediction']

# Perform one-hot encoding on the features
X = pd.get_dummies(X, drop_first=True)

# Convert Pandas DataFrames to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the KNN classifier with n_neighbors=5
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Function to display the performance metrics and confusion matrix in a window
def show_results():
    # Display accuracy
    messagebox.showinfo("Performance Metrics", f"Accuracy: {accuracy:.2f}")

    # Display confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Create a simple GUI to display the results
root = tk.Tk()
root.title("K-Nearest Neighbors Model Performance")
root.geometry("300x150")

btn_show_results = tk.Button(root, text="Show Results", command=show_results)
btn_show_results.pack(pady=20)

root.mainloop()
