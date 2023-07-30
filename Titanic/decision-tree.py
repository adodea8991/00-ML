import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("train.csv")

# Preprocessing
# Drop irrelevant features and any rows with missing values
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df = df.dropna()

# Convert categorical features into numerical using one-hot encoding
categorical_features = ['Sex', 'Embarked']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# For the classification decision tree
X_class = df[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y_class = df['Survived']
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_class_train, y_class_train)

# Plotting the classification decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X_class.columns.tolist(), class_names=['Not Survived', 'Survived'])
plt.title("Classification Decision Tree", fontsize=16)
plt.savefig("classification_tree.png")

# Calculating the accuracy on the test set
y_class_pred = clf.predict(X_class_test)
accuracy_class = accuracy_score(y_class_test, y_class_pred)
print(f"Classification Decision Tree Accuracy: {accuracy_class:.2f}")

# Confusion matrix for the classification decision tree
conf_matrix = confusion_matrix(y_class_test, y_class_pred)
print("Confusion Matrix:")
print(conf_matrix)

# For the regression decision tree
X_reg = df[['Pclass', 'Sex_male', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y_reg = df['Age']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_reg_train, y_reg_train)

# Plotting the regression decision tree
plt.figure(figsize=(15, 10))
plot_tree(reg, filled=True, feature_names=X_reg.columns.tolist())
plt.title("Regression Decision Tree", fontsize=16)
plt.savefig("regression_tree.png")

# Calculating the mean squared error on the test set
y_reg_pred = reg.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Regression Decision Tree Mean Squared Error: {mse:.2f}")

# GUI performance output
root = tk.Tk()
root.withdraw()

msg = f"Classification Decision Tree Accuracy: {accuracy_class:.2f}\nRegression Decision Tree Mean Squared Error: {mse:.2f}"
messagebox.showinfo("Performance Output", msg)

root.mainloop()
