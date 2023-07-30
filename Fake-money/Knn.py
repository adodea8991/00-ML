import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_squared_error

# Read the data from the CSV file
data = pd.read_csv('fake_bills.csv', delimiter=';')

# Separate features (X) and target (y) for classification
X_class = data.drop(columns=['is_genuine'])
y_class = data['is_genuine']

# Separate features (X) and target (y) for regression
X_reg = data.drop(columns=['diagonal'])
y_reg = data['diagonal']

# Handle missing values with SimpleImputer for classification
imputer_class = SimpleImputer(strategy='mean')
X_class_imputed = imputer_class.fit_transform(X_class)

# Split the data into training and testing sets for classification (70-30 split)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class_imputed, y_class, test_size=0.3, random_state=42
)

# Classification KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_class_train, y_class_train)
class_predictions = knn_classifier.predict(X_class_test)

# Classification Report
print("Classification Report:")
print(classification_report(y_class_test, class_predictions))

# Plot the Classification KNN Model
plt.figure(figsize=(8, 6))
sns.scatterplot(x='diagonal', y='height_left', data=data, hue='is_genuine', palette='coolwarm', s=100)
sns.scatterplot(x=X_class_test[:, 0], y=X_class_test[:, 2], hue=class_predictions, marker='X', s=200, palette='Set1')
plt.title('Classification KNN Model')
plt.xlabel('Diagonal')
plt.ylabel('Height Left')
plt.legend(title='Is Genuine', loc='upper left', labels=['False', 'True', 'Predicted False', 'Predicted True'])
plt.show()

# Handle missing values with SimpleImputer for regression
imputer_reg = SimpleImputer(strategy='mean')
X_reg_imputed = imputer_reg.fit_transform(X_reg)

# Split the data into training and testing sets for regression (70-30 split)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_imputed, y_reg, test_size=0.3, random_state=42
)

# Regression KNN model
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_reg_train, y_reg_train)
reg_predictions = knn_regressor.predict(X_reg_test)

# Regression Mean Squared Error
reg_mse = mean_squared_error(y_reg_test, reg_predictions)
print("Regression Mean Squared Error:", reg_mse)

# Plot the Regression KNN Model
plt.figure(figsize=(8, 6))
sns.scatterplot(x='diagonal', y='height_left', data=data, color='blue', s=100)
sns.scatterplot(x=X_reg_test[:, 1], y=reg_predictions, color='red', marker='X', s=100)
plt.title('Regression KNN Model')
plt.xlabel('Height Left')
plt.ylabel('Diagonal')
plt.legend(title='Is Genuine', loc='upper left', labels=['True Bills', 'Predicted Diagonal'])
plt.show()
