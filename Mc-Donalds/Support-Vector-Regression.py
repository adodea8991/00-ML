import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the cleaned data
data_with_location = pd.read_csv("clean_data.csv")

# Step 2: Check if "review_content" column is present, if not, use "review" column
if "review_content" in data_with_location.columns:
    review_column = "review_content"
else:
    review_column = "review"

# Step 3: Convert the "review_content" column to string data type
data_with_location[review_column] = data_with_location[review_column].astype(str)

# Step 4: Fill missing values with an empty string
data_with_location[review_column].fillna("", inplace=True)

# Step 5: Split the data into features (X) and target (y)
X = data_with_location[[review_column, 'latitude', 'longitude']]
y = data_with_location['rating']

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Step 7: Build and train the first model (Linear Support Vector Regression)
model1 = make_pipeline(TfidfVectorizer(), LinearSVR())
model1.fit(X_train[review_column], y_train)

# Step 8: Make predictions on the test set
y_pred_model1 = model1.predict(X_test[review_column])

# Step 9: Evaluate the performance of the first model
mse_model1 = mean_squared_error(y_test, y_pred_model1)
r2_model1 = r2_score(y_test, y_pred_model1)

# Step 10: Print the performance of the first model
print("Model 1 Performance:")
print("Mean Squared Error:", mse_model1)
print("R-squared:", r2_model1)

# Step 11: Build and train the second model (Random Forest Regressor)
model2 = make_pipeline(TfidfVectorizer(), RandomForestRegressor())
model2.fit(X_train[review_column], y_train)

# Step 12: Make predictions on the test set
y_pred_model2 = model2.predict(X_test[review_column])

# Step 13: Evaluate the performance of the second model
mse_model2 = mean_squared_error(y_test, y_pred_model2)
r2_model2 = r2_score(y_test, y_pred_model2)

# Step 14: Print the performance of the second model
print("\nModel 2 Performance:")
print("Mean Squared Error:", mse_model2)
print("R-squared:", r2_model2)

# Step 15: Visualize the actual vs. predicted ratings for both models
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_model1, alpha=0.5)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Model 1: Actual vs. Predicted Ratings')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_model2, alpha=0.5)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Model 2: Actual vs. Predicted Ratings')

plt.tight_layout()
plt.show()
