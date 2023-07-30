import pandas as pd

# Load the data from 'emails.csv'
data = pd.read_csv('emails.csv')
data.columns = data.columns.str.strip()  # Remove leading/trailing whitespaces from column names

# Print the column names to inspect them
print("Column Names:")
print(data.columns)

# Rest of the code...
