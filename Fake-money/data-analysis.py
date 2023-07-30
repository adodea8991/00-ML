import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('fake_bills.csv', delimiter=';')

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Missing Data: Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

# Plot Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='diagonal', y='height_left', data=data, hue='is_genuine', palette='coolwarm', s=100)
plt.title('Scatterplot: Diagonal vs. Height Left')
plt.xlabel('Diagonal')
plt.ylabel('Height Left')
plt.legend(title='Is Genuine', loc='upper left', labels=['False', 'True'])
plt.show()
