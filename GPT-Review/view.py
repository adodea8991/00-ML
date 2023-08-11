import pandas as pd

# Load the dataset
data = pd.read_csv('chatgpt_reviews.csv')

# Remove the 'date' column
data_view = data.drop(columns=['date'])

# Display the first few rows of the data view
print(data_view.head())

# Get the number of rows in the dataset
num_rows = data.shape[0]
print("Number of rows:", num_rows)

# Check for missing data
missing_data = data.isnull().sum()
print("Missing data:\n", missing_data)
