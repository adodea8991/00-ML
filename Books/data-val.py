import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv("main_dataset.csv")

# Interpolate missing values in the "price" column
data["price"] = pd.to_numeric(data["price"], errors="coerce")
data["price"] = data["price"].interpolate()

# Ensure all resulting values are positive
data["price"] = data["price"].apply(lambda x: max(x, 0))

# Keep only one decimal point
data["price"] = data["price"].apply(lambda x: round(x, 1))

# Extract the columns for the plots
book_depository_stars = data["book_depository_stars"]
price = data["price"]
format_data = data["format"]

# Scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(price, book_depository_stars, alpha=0.6)
plt.xlabel('Price')
plt.ylabel('Book Depository Stars')
plt.title('Scatterplot of Price vs. Book Depository Stars')
plt.show()

# Histogram of Book Depository Stars
plt.figure(figsize=(8, 6))
plt.hist(book_depository_stars, bins=20, edgecolor='black')
plt.xlabel('Book Depository Stars')
plt.ylabel('Frequency')
plt.title('Histogram of Book Depository Stars')
plt.show()

# Box Plot of Price
plt.figure(figsize=(8, 6))
plt.boxplot(price)
plt.ylabel('Price')
plt.title('Box Plot of Price')
plt.show()
