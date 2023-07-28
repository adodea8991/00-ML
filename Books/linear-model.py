import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("clean_data.csv")

# Remove any rows with missing target values (price)
df.dropna(subset=["price"], inplace=True)

# Interpolate missing values in 'old_price' column
df["old_price"].interpolate(method="pad", inplace=True)

# Handle missing values in 'format' column by forward filling (pad)
df["format"].fillna(method="pad", inplace=True)

# Label encode the 'format' column
format_encoder = LabelEncoder()
df["format_encoded"] = format_encoder.fit_transform(df["format"])

# Remove unnecessary columns
df.drop(columns=["image", "name", "author", "category", "format"], inplace=True)

# Prepare the data
X = df.drop(columns=["price"])
y = df["price"]

# Split the data into training and testing sets
train_size = int(0.8 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Create and fit the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
