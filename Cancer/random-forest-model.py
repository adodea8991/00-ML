import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Handle missing values (replace NaNs with the mean of each feature)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance of the Random Forest classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualize feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Visualize scores
scores = pd.DataFrame(class_report).transpose()
plt.figure(figsize=(8, 6))
sns.barplot(x=scores.index, y=scores['f1-score'])
plt.title('Model Scores')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1.0)
plt.show()

# Label the data based on predictions
X_test_labeled = X_test.copy()
X_test_labeled['Label'] = y_pred

# Visualize the labeled data
plt.figure(figsize=(8, 6))
sns.scatterplot(x='id', y='Label', data=X_test_labeled, hue='diagnosis', palette='coolwarm', alpha=0.8)
plt.title('True and Predicted Diagnoses')
plt.xlabel('ID')
plt.ylabel('Diagnosis')
plt.legend(title='True Diagnosis', loc='upper left', labels=['Benign', 'Malignant'])
plt.show()
