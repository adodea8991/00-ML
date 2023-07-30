import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# Step 1: Load the data from 'output.csv'
data = pd.read_csv('output.csv')

# Step 2: Split data into features (X) and target (y)
X = data.drop(columns=['Freshness', 'Image'])
y = data['Freshness']

# Step 3: Split dataset into training and testing sets (75-25 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Create SVM classifier and train on the training data
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 5: Evaluate the model's performance on the test data
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Step 6: Display the performance metrics in a GUI window
root = tk.Tk()
root.title("SVM Model Performance")
root.geometry("300x200")

accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}")
accuracy_label.pack(pady=10)

conf_matrix_label = tk.Label(root, text="Confusion Matrix:")
conf_matrix_label.pack()

conf_matrix_text = tk.Text(root, height=3, width=15)
conf_matrix_text.insert(tk.END, str(conf_matrix))
conf_matrix_text.pack(pady=10)

messagebox.showinfo("Model Performance", f"Accuracy: {accuracy:.2f}\n\nConfusion Matrix:\n{conf_matrix}")

root.mainloop()
