import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv("main_dataset.csv")

# Step 2: Create a boolean DataFrame representing missing values
missing_values = df.isnull()

# Step 3: Use seaborn to create the heatmap for missing values
plt.figure(figsize=(10, 6))
sns.heatmap(missing_values, cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Additional View 1: Book Depository Stars per category
average_stars_per_category = df.groupby("category")["book_depository_stars"].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=average_stars_per_category.index, y=average_stars_per_category.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Category")
plt.ylabel("Average Book Depository Stars")
plt.title("Average Book Depository Stars per Category")
plt.show()

# Additional View 2: Box plot of "price" based on "category" (taking currency into account)
# Clean "price" column by removing non-numeric characters
df["price"] = df["price"].str.replace(r'[^\d.]+', '', regex=True).astype(float)

plt.figure(figsize=(10, 6))
sns.boxplot(x="category", y="price", hue="currency", data=df)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Category")
plt.ylabel("Price")
plt.title("Box Plot of Price based on Category")
plt.legend(title="Currency")
plt.show()

# Additional View 3: Average delta between "price" and "old_price" per category
# Clean "old_price" column by removing non-numeric characters and replace empty strings with NaN
df["old_price"] = df["old_price"].replace('', pd.NA).str.replace(r'[^\d.]+', '', regex=True).astype(float)

# Calculate the "price_delta" by filling NaN values with zeros
df["price_delta"] = df["old_price"].fillna(0) - df["price"]
average_delta_per_category = df.groupby("category")["price_delta"].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=average_delta_per_category.index, y=average_delta_per_category.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Category")
plt.ylabel("Average Price Delta")
plt.title("Average Delta between Price and Old Price per Category")
plt.show()
