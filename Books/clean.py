import pandas as pd

def clean_data(input_file, output_file):
    # Step 1: Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Step 2: Fill empty cells in the "author" column with "Missing"
    df["author"].fillna("Missing", inplace=True)

    # Step 3: Clean "price" column by removing non-numeric characters and convert to float
    df["price"] = df["price"].str.replace(r'[^\d.]+', '', regex=True).astype(float)

    # Step 4: Interpolate the values in the "old_price" column
    df["old_price"] = df["old_price"].interpolate(method='linear')

    # Step 5: Export the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = "main_dataset.csv"
    output_file = "clean_data.csv"
    clean_data(input_file, output_file)
