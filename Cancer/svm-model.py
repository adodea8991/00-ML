import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data from data.csv
data_path = "data.csv"
data = pd.read_csv(data_path)

# Drop the 'Unnamed: 32' column if it exists
if 'Unnamed: 32' in data.columns:
    data.drop(columns=['Unnamed: 32'], inplace=True)

# Preprocess the data
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Encode the target variable 'diagnosis' (Malignant = 1, Benign = 0)
y = y.map({'M': 1, 'B': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_test_pred)

# Plot the overall performance of the model
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy])
plt.ylim(0, 1)
plt.title('Model Performance')
plt.ylabel('Accuracy')
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from data.csv
data_path = "data.csv"
data = pd.read_csv(data_path)

# Drop the 'Unnamed: 32' column if it exists
if 'Unnamed: 32' in data.columns:
    data.drop(columns=['Unnamed: 32'], inplace=True)

# Preprocess the data
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Encode the target variable 'diagnosis' (Malignant = 1, Benign = 0)
y = y.map({'M': 1, 'B': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_test_pred)

# Print relevant information about the model's performance
print(f"Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
