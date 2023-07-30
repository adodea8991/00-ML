import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# Load the data from 'emails.csv'
data = pd.read_csv('emails.csv')
data.columns = data.columns.str.strip()  # Remove leading/trailing whitespaces from column names

# Set 'Prediction' as the target variable
y = data['Prediction']

# Drop the target variable and the 'Email No.' column (assuming it's not needed for prediction)
X = data.drop(['Email No.', 'Prediction'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a GUI to display the Random Forest performance, including the confusion matrix
root = tk.Tk()
root.title("Random Forest Performance")
root.geometry("300x200")

accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}")
accuracy_label.pack(pady=10)

# Display the confusion matrix in a label
cm_label = tk.Label(root, text="Confusion Matrix:")
cm_label.pack()

# Create a string representation of the confusion matrix
cm_str = "\n".join([" ".join(map(str, row)) for row in cm])

# Display the confusion matrix in a multiline label
cm_display = tk.Label(root, text=cm_str, justify='left', anchor='w', padx=10)
cm_display.pack()

root.mainloop()
