import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(df):
    # Drop rows with missing values in column D
    df.dropna(subset=['D'], inplace=True)
    # Drop unnecessary columns from J onwards
    df.drop(df.columns[9:], axis=1, inplace=True)
    return df

def train_knn_regression(train_data, train_labels, k):
    # Create and train the KNN regression model
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(train_data, train_labels)
    return model

def visualize_model(model, test_data, test_labels):
    # Make predictions on the test data
    predictions = model.predict(test_data)

    # Create a scatter plot to visualize the model's predictions vs. actual values
    plt.scatter(test_labels, predictions, color='blue', alpha=0.5)
    plt.scatter(test_labels, test_labels, color='red', alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("KNN Regression Model")
    plt.legend(["Predicted", "Actual"])
    plt.show()

def evaluate_model(model, test_data, test_labels):
    # Make predictions on the test data
    predictions = model.predict(test_data)

    # Calculate mean squared error and R-squared score
    mse = mean_squared_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)

    return mse, r2

def main():
    # Load the datasets with specified data types for columns with mixed types
    training_data = pd.read_csv('training.csv', dtype={'J': str, 'L': str, 'N': str, 'A': float, 'B': float, 'C': float, 'D': float, 'E': float, 'F': float, 'G': float, 'H': float}, low_memory=False)
    test_data = pd.read_csv('test.csv', dtype={'J': str, 'L': str, 'N': str, 'A': float, 'B': float, 'C': float, 'D': float, 'E': float, 'F': float, 'G': float, 'H': float}, low_memory=False)

    # Preprocess the data for training and testing
    training_data = preprocess_data(training_data)
    test_data = preprocess_data(test_data)

    # Separate the features and labels for training data
    train_features = training_data.drop(columns=['D'])
    train_labels = training_data['D']

    # Train the KNN regression model with k=5 (you can adjust k as needed)
    k = 5
    model = train_knn_regression(train_features, train_labels, k)

    # Visualize the model on the test dataset
    visualize_model(model, test_data.drop(columns=['D']), test_data['D'])

    # Evaluate the model on the test dataset
    mse, r2 = evaluate_model(model, test_data.drop(columns=['D']), test_data['D'])
    print("Performance on Test Dataset:")
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

    # Create a simple pop-up window to show the model's performance
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"Mean Squared Error: {round(mse, 2)}\nR-squared Score: {round(r2, 2)}", ha='center', va='center', fontsize=12)
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()
