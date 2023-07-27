import pandas as pd
from datetime import datetime, timedelta

# Step 1: Load the data with the appropriate encoding (utf-8)
try:
    data = pd.read_csv("McD_Reviews.csv", encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv("McD_Reviews.csv", encoding='latin-1')

# Step 2: Convert the review_time column to datetime
def parse_relative_time(relative_time):
    if 'months ago' in relative_time:
        months_ago = int(relative_time.split()[0])
        return datetime.now() - timedelta(days=months_ago*30)
    elif 'days ago' in relative_time:
        days_ago = int(relative_time.split()[0])
        return datetime.now() - timedelta(days=days_ago)
    else:
        return pd.NaT

data["review_time"] = data["review_time"].apply(parse_relative_time)

# Step 3: Clean the column names by removing whitespaces
data.columns = data.columns.str.strip()

# Step 4: Drop unnecessary columns if they exist
if 'restaurant' in data.columns and 'city' in data.columns:
    data = data.drop(columns=['restaurant', 'city'])

# Step 5: Drop rows with missing latitude and longitude values
data_with_location = data.dropna(subset=['latitude', 'longitude'])

# Step 6: Preprocess the rating column
def parse_rating(rating):
    return int(rating.split()[0])

data_with_location['rating'] = data_with_location['rating'].apply(parse_rating)

# Step 7: Remove instances of invalid characters
data_with_location = data_with_location.replace({r'[^\x00-\x7F]+': ''}, regex=True)

# Step 8: Export the cleaned data to a CSV file
data_with_location.to_csv("clean_data.csv", index=False)

# Print the first few rows of the cleaned data
print(data_with_location.head())
