import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data from the .csv file
data = pd.read_csv('house_data_generated.csv')

# Removing outliers using z-score method
z_scores = (data - data.mean()) / data.std()
data = data[(z_scores < 3).all(axis=1)]

# Data normalization using min-max scaling
scaler = StandardScaler()
data[['House_Size', 'House_Price']] = scaler.fit_transform(data[['House_Size', 'House_Price']])

# Ensure all data is positive by taking the absolute value
data = data.abs()

# Round the data to one decimal point
data = data.round(1)

# Data split into training and test sets (70% training, 30% test)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Save the cleaned data to 'clean_data.csv'
train_data.to_csv('clean_data.csv', index=False)

# Display the cleaned data
print(train_data.head())
