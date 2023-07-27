import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# Step 1: Load the data with the appropriate encoding
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

# Step 4: Drop rows with missing latitude and longitude values
data_with_location = data.dropna(subset=['latitude', 'longitude'])

# Step 5: Preprocess the rating column
def parse_rating(rating):
    return int(rating.split()[0])

data_with_location['rating'] = data_with_location['rating'].apply(parse_rating)

# Step 6: Perform KMeans clustering on latitude, longitude, and rating
X = data_with_location[['latitude', 'longitude', 'rating']]
kmeans = KMeans(n_clusters=5, random_state=42)
data_with_location['cluster'] = kmeans.fit_predict(X)

# Step 7: Plot the clusters
sns.scatterplot(data=data_with_location, x='longitude', y='latitude', hue='cluster', palette='tab10', s=100)
plt.title('McDonald\'s Reviews - Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
