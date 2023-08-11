import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('chatgpt_reviews.csv')

# Remove the 'date' column
data = data.drop(columns=['date'])

# Split the data into training, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the split data into separate CSV files
train_data.to_csv('training.csv', index=False)
valid_data.to_csv('validation.csv', index=False)
test_data.to_csv('test.csv', index=False)
