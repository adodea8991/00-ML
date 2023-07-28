import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_csv(file_path):
    return pd.read_csv(file_path)

def compare_csv(data_csv, output_csv):
    df_data = read_csv(data_csv)
    df_output = read_csv(output_csv)

    # Find differences
    differences = df_data.compare(df_output)

    # Generate heatmap
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(differences.abs(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.title("Differences between data.csv and output.csv")
    plt.xlabel("Columns")
    plt.ylabel("Index")
    plt.tight_layout()

    # Add colorbar title
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Absolute Difference")

    plt.savefig("heatmap.png")
    plt.show()

if __name__ == "__main__":
    data_csv = "data.csv"
    output_csv = "output.csv"
    compare_csv(data_csv, output_csv)
