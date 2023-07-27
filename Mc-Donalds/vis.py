import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
data = pd.read_csv('clean_data.csv')

# Plot the histogram of ratings
plt.figure(figsize=(8, 6))
sns.histplot(data['rating'], bins=10, kde=True, color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of McDonald\'s Ratings')
plt.show()

# Plot the bar plot of average ratings by store address
plt.figure(figsize=(12, 6))
average_ratings = data.groupby('store_address')['rating'].mean().reset_index()
sns.barplot(x='rating', y='store_address', data=average_ratings, color='lightgreen')
plt.xlabel('Average Rating')
plt.ylabel('Store Address')
plt.title('Average Ratings by Store Address')
plt.show()
