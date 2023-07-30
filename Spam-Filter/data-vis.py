import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('emails.csv')

# Check the first few rows of the data
print(data.head())

# Count the occurrences of each column in the data
column_counts = data.count()

# Visualize the column occurrences
plt.figure(figsize=(12, 6))
sns.barplot(x=column_counts.index, y=column_counts.values, palette='viridis')
plt.xticks(rotation=90)
plt.xlabel('Columns')
plt.ylabel('Count')
plt.title('Occurrences of Columns in the Dataset')
plt.show()
