import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import plot_tree

# Load data
data = pd.read_csv('train.csv')

# Drop rows with missing values in 'Age' column
data = data.dropna(subset=['Age'])

# Select features and target for classification
X_class = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_class = data['Survived']

# Convert categorical variables to numeric using one-hot encoding
X_class = pd.get_dummies(X_class, drop_first=True)

# Split data into training and testing sets for classification
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train the classification decision tree
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_class_train, y_class_train)

# Predict on the test set
y_class_pred = clf.predict(X_class_test)

# Calculate classification accuracy
class_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"Classification Decision Tree Accuracy: {class_accuracy:.2f}")

# Plot the classification decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X_class.columns.tolist(), class_names=['Not Survived', 'Survived'])
plt.title("Classification Decision Tree")
plt.show()

# Select features and target for regression
X_reg = data[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_reg = data['Age']

# Convert categorical variables to numeric using one-hot encoding
X_reg = pd.get_dummies(X_reg, drop_first=True)

# Split data into training and testing sets for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train the regression decision tree
reg = DecisionTreeRegressor(criterion='friedman_mse')
reg.fit(X_reg_train, y_reg_train)

# Predict on the test set
y_reg_pred = reg.predict(X_reg_test)

# Calculate mean squared error for regression
reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Regression Decision Tree Mean Squared Error: {reg_mse:.2f}")

# Plot the regression decision tree
plt.figure(figsize=(15, 10))
plot_tree(reg, filled=True, feature_names=X_reg.columns.tolist())
plt.title("Regression Decision Tree")
plt.show()

# Visualization of impurity measures

# Generate class proportions from 0 to 1
p_values = np.linspace(0, 1, num=100)
gini_values = 1 - (p_values**2 + (1-p_values)**2)
entropy_values = - (p_values * np.log2(p_values) + (1-p_values) * np.log2(1-p_values))

# Plot impurity measures
plt.figure(figsize=(10, 6))
plt.plot(p_values, gini_values, label='Gini Index')
plt.plot(p_values, entropy_values, label='Entropy')
plt.xlabel('Proportion of Class 1')
plt.ylabel('Impurity')
plt.title('Impurity Measures Comparison')
plt.legend()
plt.grid()
plt.show()
