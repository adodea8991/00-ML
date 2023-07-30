import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_plots(csv_file):
    # Read data from CSV file, specifying the data types for problematic columns
    dtypes = {'MARRIAGE': str, 'EDUCATION': str, 'PAY_0': str, 'PAY_2': str, 'PAY_3': str,
              'PAY_4': str, 'PAY_5': str, 'PAY_6': str}
    df = pd.read_csv(csv_file, dtype=dtypes)
    
    # Drop irrelevant columns with mixed types
    df = df.drop(columns=['ID'])
    
    # Convert non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(f"Heatmap - {csv_file}")
    plt.savefig(f"{csv_file.split('.')[0]}_heatmap.png")
    plt.close()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    df.hist()
    plt.suptitle(f"Histograms - {csv_file}", fontsize=16)
    plt.savefig(f"{csv_file.split('.')[0]}_histogram.png")
    plt.close()
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title(f"Box Plot - {csv_file}")
    plt.savefig(f"{csv_file.split('.')[0]}_boxplot.png")
    plt.close()

# Create plots for test.csv and training.csv
create_plots("test.csv")
create_plots("training.csv")
