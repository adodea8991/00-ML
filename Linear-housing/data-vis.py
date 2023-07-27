import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the .csv file
data = pd.read_csv('house_data_generated.csv')

# Scatter plot for 'House_Size' vs 'House_Price'
plt.figure(figsize=(10, 6))
plt.scatter(data['House_Size'], data['House_Price'], alpha=0.5)
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('House Price vs House Size')

# Box plot for 'House_Price'
plt.figure(figsize=(8, 6))
sns.boxplot(data['House_Price'])
plt.xlabel('House Price')
plt.title('Box Plot of House Price')

# Violin plot for 'House_Size'
plt.figure(figsize=(8, 6))
sns.violinplot(data['House_Size'])
plt.xlabel('House Size')
plt.title('Violin Plot of House Size')

plt.show()