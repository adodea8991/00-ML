# Titanic Data Analysis Project

![Missing data](https://github.com/adodea8991/00-ML/blob/main/Titanic/Screenshot%202023-07-27%20at%2023.54.23.png)

## Introduction

This is a data analysis project focused on the famous Titanic dataset. The dataset contains information about passengers on board the Titanic, including whether they survived or not, their socio-economic status, age, gender, and more. The main objective of this project is to analyze the data, gain insights, and create visualizations to better understand the factors that influenced survival rates on the Titanic.

## Dataset

The dataset is stored in the file `train.csv`, which contains the following columns:

- PassengerId: Unique identifier for each passenger
- Survived: Whether the passenger survived (1) or not (0)
- Pclass: Passenger class (1st, 2nd, or 3rd class)
- Name: Name of the passenger
- Sex: Gender of the passenger (male or female)
- Age: Age of the passenger
- SibSp: Number of siblings/spouses aboard the Titanic
- Parch: Number of parents/children aboard the Titanic
- Ticket: Ticket number
- Fare: Fare paid for the ticket
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Missing Data Analysis

In order to understand the missing data within the dataset, we have created a Python script `missing.py`. This script will load the data from `train.csv` and generate a heatmap visualization to highlight the missing data in the dataset.

### Prerequisites

To run the `missing.py` script, you need to have the following installed:

- Python 3
- Pandas
- Seaborn
- Matplotlib

### How to Use

1. Make sure you have Python 3 installed on your system.

2. Install the required libraries by running:

```bash
pip install pandas seaborn matplotlib
```

3. Clone this repository to your local machine or download the `train.csv` file.

4. Run the `missing.py` script using the following command:

```bash
python missing.py
```

5. The script will read the data from `train.csv`, create a heatmap, and display it. The heatmap will help visualize the missing data in the dataset, making it easier to identify the columns with missing values.

## Conclusion

By analyzing the Titanic dataset and visualizing the missing data, we aim to gain valuable insights into the factors that influenced survival rates. This project can serve as a starting point for further data analysis, feature engineering, and machine learning modeling for predicting survival probabilities on the Titanic.

Feel free to contribute to this project by adding more analyses, visualizations, or even implementing machine learning algorithms to predict passenger survival.

Let's explore the Titanic dataset and discover intriguing patterns! 🚢🔍




# Linear Model preview

![House Price Prediction Lieanr Model Visualisation](https://github.com/adodea8991/00-ML/blob/main/Linear-housing/Linear_regression.png)


## Overview

This repository contains a Python implementation of a linear regression model for predicting house prices based on various features. The model is trained on a dataset of houses with their corresponding prices and features such as house size, number of bedrooms, and number of bathrooms.

## Prerequisites
Before running the code, ensure you have the following dependencies installed:

-Python (version 3.x)

-NumPy

-pandas

-matplotlib

-scikit-learn

You can install these dependencies using pip:

```python
python3 linear_regression_model.py
```


Dataset
The dataset used for this project is stored in the "data" folder, which contains two CSV files:

train.csv: This file contains the training data, which includes the features (X) and the target variable (Y) for house prices.
test.csv: This file contains the test data, where you can evaluate the model's performance after training.
Running the Model
To train and test the linear regression model, run the linear_regression_model.py script:

```python
python3 linear_regression_model.py
```

The script will load the training and test data from the CSV files, preprocess the data, and fit the linear regression model using scikit-learn. The model's performance metrics, such as Mean Squared Error (MSE) and R-squared, will be displayed in the console.

Model Evaluation
The linear regression model's performance can be evaluated using various metrics, including:

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual house prices. Lower values indicate better performance.
R-squared (Coefficient of Determination): Represents the proportion of variance in the target variable explained by the model. A value close to 1 indicates a good fit.
Results
After running the model, you can find the evaluation results in the console. Additionally, the script will generate visualizations of the actual house prices against the predicted prices, allowing you to observe how well the model fits the data.

Interpreting the Model
The linear regression model uses a straight line to represent the relationship between house features and prices. The equation of the line is:

Y = mx + b
Where:

Y is the predicted house price.
m is the coefficient (slope) of the line for each feature.
x is the value of the feature.
b is the y-intercept of the line.
By examining the coefficients of the model, you can understand the impact of each feature on the predicted house price. Positive coefficients indicate a positive relationship, while negative coefficients indicate a negative relationship.

Improving the Model
To improve the model's performance, you can consider the following steps:

Feature Engineering: Experiment with additional features or transformations of existing features to capture more information about the houses.
Data Cleaning: Handle outliers and missing values in the dataset to ensure the model's robustness.
Polynomial Regression: Try fitting a polynomial regression model to capture non-linear relationships between features and house prices.
Regularization: Implement regularization techniques like Lasso or Ridge regression to prevent overfitting.
Conclusion
This project demonstrates a simple linear regression model for house price prediction. By understanding the model's performance metrics and visualizations, you can assess its effectiveness in predicting house prices based on the given features. Use this repository as a starting point to explore more advanced machine learning algorithms and improve your understanding of predictive modeling.