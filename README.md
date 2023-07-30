# Machine Learning Projects Portofolio

## Table of Contents

1. [Credit Classification Project using Multiple Algorithms](#credit-classification-project)
2. [Cancer Diagnosis using Machine Learning](#cancer-diagnosis-using-machine-learning)
3. [Book Price Prediction Model](#book-price-prediction-model)
4. [K-Nearest Neighbour Fake Bills Classification and Regression](#knn-fake-bills-classification-and-regression)
5. [McDonald's Rating Prediction Project](#mcdonalds-rating-prediction-project)
6. [Titanic Data Analysis Project](#titanic-data-analysis-project)
7. [Spam Filter using Multiple Algorithms](#spam-filter-multiple-algorithms)
8. [Housing Linear Model](#housing-linear-model)





## Credit-Classification-Project

This repository contains code for a machine learning project that aims to predict credit classification using various regression algorithms. The project involves data pre-processing, training and evaluating linear regression, decision tree, and K-Nearest Neighbors (KNN) regression models, and visualizing their performance.

### Overview

The main goal of this project is to develop regression models that can predict credit classification based on various features in the dataset. We will use the following regression algorithms:

1. Linear Regression
2. Decision Tree Regression
3. K-Nearest Neighbors (KNN) Regression

The dataset used for this project contains information about credit applicants and their respective credit classification. The dataset requires pre-processing due to some missing values and mixed data types in certain columns.

### Step-by-Step Implementation

### 1. Data Pre-processing


![Training Heatmap](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/training_heatmap.png)
![Training Histogram](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/training_histogram.png)
![Training Boxplot](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/training_boxplot.png)


The dataset contains missing values and columns with mixed data types. The following steps are performed for data pre-processing:

- Drop rows with missing values in the target column (D).
- Drop unnecessary columns (columns J onwards).
- Encode categorical features (column I) using one-hot encoding.

![Test Heatmap](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/test_heatmap.png)
![Test Histogram](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/test_histogram.png)
![Test Boxplot](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/test_boxplot.png)

### 2. Linear Regression

![Linear Regression Model](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/Linear-model.png)
![Linear Regression Performance](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/Linear-performance.png)


The linear regression algorithm is implemented to predict the credit classification. We use the following steps:

- Separate the features and labels from the pre-processed data.
- Train the linear regression model on the training data.
- Evaluate the model's performance on the test data using Mean Squared Error and R-squared score.
- Visualize the model's performance using a scatter plot of predicted vs. actual values.

### 3. Decision Tree Regression

![Decision Tree Model](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/Decision-model.png)
![Decision Tree Model Performance](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/Decision-tree-performance.png)

The decision tree regression algorithm is implemented to predict credit classification. The steps include:

- Separate the features and labels from the pre-processed data.
- Train the decision tree regression model on the training data.
- Evaluate the model's performance on the test data using Mean Squared Error and R-squared score.
- Visualize the model's performance using a scatter plot of predicted vs. actual values.
- Display the decision tree structure.

### 4. K-Nearest Neighbors (KNN) Regression

![KNN Model](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/knn-model.png)
![KNN Model Performance](https://github.com/adodea8991/00-ML/blob/main/Credit-Classification/knn-performance.png)

The KNN regression algorithm is implemented to predict credit classification. The steps include:

- Separate the features and labels from the pre-processed data.
- Train the KNN regression model on the training data with a chosen value of k.
- Evaluate the model's performance on the test data using Mean Squared Error and R-squared score.
- Visualize the model's performance using a scatter plot of predicted vs. actual values.
- Display the KNN model's performance in a pop-up window.

### Visualizations

The following visualizations are included in the project:

1. Linear Regression Model: A scatter plot of predicted vs. actual values for the test dataset.
2. Decision Tree Model: A scatter plot of predicted vs. actual values for the test dataset.
3. KNN Regression Model: A scatter plot of predicted vs. actual values for the test dataset.

### Conclusion

Through this project, we explored the application of three regression algorithms for credit classification prediction. The decision tree and KNN regression models showed better performance compared to linear regression, with higher R-squared scores and lower Mean Squared Error.

This project demonstrates the importance of data pre-processing and model evaluation in machine learning tasks. Additionally, the visualization of model performance provides insights into the model's effectiveness in predicting credit classification. We've learned some models fair VERY DIFFERENTLY from others and it's highly important to cross-validate models.


### Dependencies
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn




## Cancer-Diagnosis-Using-Machine-Learning

### Overview

This project focuses on predicting cancer diagnosis (Malignant or Benign) using machine learning techniques. The dataset used in this project is from the 'data.csv' file, which contains various features related to breast cancer patients. The primary objective is to develop accurate models that can assist in early cancer detection and improve patient outcomes.


### Dataset

The dataset we used for this project is stored in a file called `data.csv`. It contains various features related to cell characteristics, such as radius_mean, texture_mean, perimeter_mean, area_mean, etc., along with the diagnosis (target variable). Before starting the analysis and modeling, we performed initial data exploration to gain insights into the dataset. As part of this exploration, we created a heatmap to understand the correlations between different features in the dataset.

### Data Exploration

During the data exploration phase, we first loaded the dataset into a pandas DataFrame. Since the diagnosis column contained non-numeric values ('M' and 'B'), we converted it into binary numeric values. 'M' was mapped to 1 (malignant) and 'B' to 0 (benign).

Next, we created a correlation matrix using the `corr()` function from pandas to measure the linear relationships between numeric features in the dataset. The correlation matrix provides valuable insights into how features are related to each other, which is crucial for feature selection and model building.

### Correlation Heatmap

To visualize the correlation matrix, we used the seaborn library to create a heatmap. The heatmap helps us visually identify the strength and direction of the relationships between features. Positive values indicate positive correlations, while negative values indicate negative correlations. A value close to 1 or -1 indicates a strong correlation, while values close to 0 indicate a weak or no correlation.

The correlation heatmap allowed us to understand which features are highly correlated and potentially redundant, as well as which features are highly correlated with the target variable (diagnosis). By analyzing the heatmap, we gained insights into the interdependencies among the features, which informed our feature selection and model building decisions.

![Parameter Heatmap](https://github.com/adodea8991/00-ML/blob/main/Cancer/Param-Heatmap.png)




**Implementation:**

The implementation was done in Python using the NumPy library for numerical computations, pandas for data analysis, and scikit-learn for dataset loading and train-test splitting. We performed the following steps:

1. Load the dataset and preprocess it by standardizing the features using `StandardScaler`.
2. Add a bias term to the features matrix to account for the intercept term in logistic regression.
3. Split the data into training and test sets using `train_test_split`.
4. Define the logistic function (sigmoid function), cost function with L2 regularization, and gradient descent algorithm for logistic regression.
5. Train the logistic regression model on the training data using gradient descent.
6. Make predictions on the test set and evaluate the model's accuracy.
7. Compute and print the confusion matrix to understand the model's performance.

**Reasoning:**

We chose to implement logistic regression for this classification task because it is a simple and effective algorithm for binary classification problems. Logistic regression works well when the data is linearly separable and provides interpretable probabilities for class predictions.

Moreover, we incorporated L2 regularization (ridge regularization) into the cost function to prevent overfitting and improve the model's generalization ability.

**Results and Learnings:**

After implementing the logistic regression model, we achieved an accuracy of approximately 99%, which indicates that the model is performing well on the test set. The confusion matrix revealed that the model made very few misclassifications. It correctly predicted 107 malignant tumors (true positives) and 62 benign tumors (true negatives). There was only one false positive and one false negative, indicating a high level of precision and recall.


![Confusion Matrix](https://github.com/adodea8991/00-ML/blob/main/Cancer/Confusion-matrix.png)
![Cost Function](https://github.com/adodea8991/00-ML/blob/main/Cancer/Cost-function.png)





### Model Selection

Two other machine learning models were chosen for this project:

1. **Random Forest:** Random Forest is an ensemble learning method that constructs multiple decision trees and combines their outputs to enhance accuracy and reduce overfitting.

2. **Support Vector Machine (SVM):** SVM is a powerful classification algorithm that finds an optimal hyperplane to separate different classes and maximize the margin between them.

### Key Steps in the Process

1. **Data Preprocessing:** The data was carefully examined for any missing values or inconsistent entries. Necessary preprocessing steps were taken to handle missing data and encode categorical variables.

2. **Train-Test Split:** The dataset was divided into a training set (75% of the data) and a test set (25% of the data) to evaluate the model's performance on unseen data.

3. **Model Training:** Both Random Forest and SVM models were trained using the training data.

4. **Model Evaluation:** The models were evaluated on the test set using various performance metrics such as accuracy, precision, recall, and F1-score. The confusion matrix was also used to visualize the models' classification performance.

### Model Performance

1. **Random Forest:**
- Accuracy: 97%
- Confusion Matrix:
```
[[87  2]
 [ 3 51]]
```
- Classification Report:
```
 {'0': {'precision': 0.9666666666666667, 'recall': 0.9775280898876404, 'f1-score': 0.9720670391061451, 'support': 89.0}, '1': {'precision': 0.9622641509433962, 'recall': 0.9444444444444444, 'f1-score': 0.9532710280373832, 'support': 54.0}, 'accuracy': 0.965034965034965, 'macro avg': {'precision': 0.9644654088050315, 'recall': 0.9609862671660424, 'f1-score': 0.9626690335717641, 'support': 143.0}, 'weighted avg': {'precision': 0.9650041782117255, 'recall': 0.965034965034965, 'f1-score': 0.9649692447165427, 'support': 143.0}}
```
- Key Takeaway: Random Forest also achieved an accuracy of 97%, with slightly lower precision and recall compared to SVM for the malignant class.

![Random Forest Scores](https://github.com/adodea8991/00-ML/blob/main/Cancer/Model-score.png)
![Feature Importance](https://github.com/adodea8991/00-ML/blob/main/Cancer/Random-forest-importance.png)









2. **Support Vector Machine (SVM):**
- Accuracy: 97%
- Classification Report:
```
              precision    recall  f1-score   support

           0       0.97      0.98      0.97        89
           1       0.96      0.94      0.95        54

    accuracy                           0.97       143
   macro avg       0.96      0.96      0.96       143
weighted avg       0.97      0.97      0.96       143
```
- Key Takeaway: SVM achieved an impressive accuracy of 97% with balanced precision and recall for both malignant and benign classes, making it a reliable model for cancer diagnosis.

![SVM performance](https://github.com/adodea8991/00-ML/blob/main/Cancer/Svm-performance.png)
![Confusion Matrix](https://github.com/adodea8991/00-ML/blob/main/Cancer/Confusion-matrix.png)


### Key Takeaways and Next Steps

1. **Feature Engineering:** Feature engineering can play a crucial role in improving model performance. Exploring new features and selecting relevant ones can lead to better predictive models.

2. **Hyperparameter Tuning:** We can perform hyperparameter tuning to find the best regularization parameter `lambda_reg` and learning rate `alpha`, which might improve the model's performance further.

3. **Ensemble Techniques:** Ensemble methods like boosting and bagging can be employed to combine multiple models for even better predictive performance.

4. **Deep Learning:** Exploring deep learning models like Neural Networks can be beneficial, especially when dealing with large-scale datasets.

5. **Visualizations:** Further exploring visualization techniques can provide valuable insights into the dataset and model decisions.

6. **Interpretability:** Considering the critical nature of cancer diagnosis, models with higher interpretability can be preferred to gain insights into the decision-making process.

7. **Cross-Validation:** Implementing k-fold cross-validation to get a more robust estimate of the model's performance and reduce the variance of the results.

8. **Other Regularization Techniques:** Experimenting with other regularization techniques like L1 regularization (Lasso) and Elastic Net regularization to compare their impact on the model's performance.


### Conclusion

This project demonstrates the potential of machine learning in cancer diagnosis using the breast cancer dataset. Both Random Forest and SVM models achieved high accuracy in predicting cancer diagnoses. SVM, with its balanced precision and recall, emerged as the better-performing model. The next steps involve further refining the models and exploring advanced techniques to improve accuracy and interpretability, ultimately contributing to more effective cancer diagnosis and treatment.





## Book-Price-Prediction-Model

This repository contains the code and data for the Book project, where we analyze and build models to predict book prices and book_depository_stars based on various features.

**Data Pre-analysis:**
Before diving into modeling, we performed data pre-analysis using various visualization techniques, including heatmaps, scatter plots, histograms, and box plots. These visualizations helped us understand the distribution of the target variable (book_depository_stars) and the features. We used heatmaps to identify any correlations between variables and scatter plots to observe relationships between book_depository_stars and other features. Histograms and box plots allowed us to analyze the spread and central tendency of the target variable and features, helping us detect potential outliers and skewed distributions.
![Heatmap Missing Data](https://github.com/adodea8991/00-ML/blob/main/Books/Missing-data.png)
![Scatterplot Price X Book Depository Stars](https://github.com/adodea8991/00-ML/blob/main/Books/Data-scatter.png)
![Histogram Book Repository Stars](https://github.com/adodea8991/00-ML/blob/main/Books/Histo.png)
![Price Box Plot](https://github.com/adodea8991/00-ML/blob/main/Books/Price-box-plot.png)



**Data Cleaning:**
To ensure the data's quality and suitability for modeling, we cleaned the data using the "clean.py" script. This script performs data cleaning tasks such as removing unnecessary columns, handling missing values, and transforming data types to appropriate formats. The cleaned data is saved as "clean_data.csv," which is used for subsequent analysis.

**Additional Visualizations:**
The "clean.py" script not only produces the cleaned dataset but also generates two insightful visualizations:
1. Average book depository stars per category: This visualization provides a quick overview of the average book_depository_stars for each category, helping us identify categories with higher or lower average ratings.
![Average book depository stars per category](https://github.com/adodea8991/00-ML/blob/main/Books/Avg-stars-category.png)

2. Box plot price per category: The box plot helps us visualize the distribution of book prices within each category, allowing us to spot differences in pricing across different book categories.
![Box plot price per category](https://github.com/adodea8991/00-ML/blob/main/Books/Box-plot-price-category.png)


**Modeling:**
For predicting book prices and book_depository_stars, we implemented two models:
1. Linear Regression: The linear regression model attempts to establish a linear relationship between the features and the target variable (price). However, the model's performance was not satisfactory, as indicated by the following metrics:
   - Mean Squared Error: 161.63086648692223
   - R-squared: 0.09936935108361933

2. Logistic Regression: The logistic regression model aims to predict the probability of book_depository_stars belonging to a certain class. However, similar to the linear regression model, it did not perform well for this dataset.

**Conclusion and Future Work:**
Based on the results obtained from the models used in this analysis, it is evident that the selected models were not appropriate for this dataset. The low R-squared and high mean squared error indicate that the models did not effectively capture the underlying relationships between the features and target variables.

To improve the analysis, we recommend exploring alternative models, such as Random Forest or Neural Networks, which may better capture the complex relationships within the data. Additionally, feature engineering and further data preprocessing could also play a significant role in improving model performance.

Overall, this project serves as a starting point for future analyses, with the potential to enhance the models and extract more valuable insights from the data.

Feel free to explore the code and data in this repository and experiment with different models and data preprocessing techniques for further improvements. Happy analyzing!

*Note: The original dataset used in this project is available in "main_dataset.csv," while the cleaned dataset is provided in "clean_data.csv." The code for data preprocessing, visualizations, and model development can be found in the respective Python scripts within the repository.*







# KNN-Fake-Bills-Classification-and-Regression

In this project, we implemented the K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks using a dataset containing information about fake bills. We'll go through each step of the project and discuss the results we obtained.

## Data Preprocessing and Initial Data Visualization

![Missing data](https://github.com/adodea8991/00-ML/blob/main/Fake-money/Missing-data.png)
![Scatterplot of true & fake bills](https://github.com/adodea8991/00-ML/blob/main/Fake-money/True-vs-Fake-scatter.png)

The dataset `fake_bills.csv` contains features such as diagonal, height_left, height_right, margin_low, margin_up, and length. The target variable is `is_genuine`, which indicates whether the bill is genuine (True) or fake (False).

Before applying KNN, we performed data preprocessing steps to handle missing values using a mean imputer from `sklearn.impute.SimpleImputer`. We then split the data into training and testing sets using a 70-30 split ratio.

To understand the data distribution and relationships between features, we created two plots: a heatmap and a scatter plot. The heatmap provided insights into the correlation between features, while the scatter plot allowed us to visualize the data points.

## K-Nearest Neighbors (KNN) Algorithm

### Classification Task

![Classification results](https://github.com/adodea8991/00-ML/blob/main/Fake-money/Classification-results.png)


For the classification task, we used `KNeighborsClassifier` from `sklearn.neighbors` to build the KNN model. The hyperparameter `n_neighbors` was set to 5, which means that the algorithm considers the 5 nearest neighbors when making predictions.

### Regression Task

![Regression Results](https://github.com/adodea8991/00-ML/blob/main/Fake-money/Knn-regression.png)

For the regression task, we used `KNeighborsRegressor` from `sklearn.neighbors` to build the KNN model. Similarly, we set `n_neighbors` to 5 for this regression model.

## Model Visualizations

For both the classification and regression tasks, we visualized the models using a GUI window created with `tkinter`. The GUI displayed the classification report and regression Mean Squared Error (MSE) after training and testing the models.

## Results and Analysis

![Decision Tree Accuracy](https://github.com/adodea8991/00-ML/blob/main/Fake-money/Accuracy.png)

### Classification Results

The classification KNN model achieved impressive performance with an accuracy of 0.99 on the test set. The precision, recall, and F1-score for both classes (True and False) were also high, indicating the model's ability to correctly classify genuine and fake bills. The macro and weighted averages were close to 0.99, indicating that the model's performance is balanced across classes.

### Regression Results

For the regression task, the KNN model achieved a Mean Squared Error (MSE) of 0.102. The lower the MSE value, the better the model's predictions match the actual target values. Therefore, the KNN regression model performed well in predicting the diagonal values of the fake bills.

## Conclusion

In this project, we successfully implemented the K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks on a dataset containing information about fake bills. The models achieved high accuracy and performed well in predicting the target variables.

The KNN algorithm is versatile and can be used for both classification and regression tasks. However, it's essential to choose an appropriate value for `n_neighbors`, as a small `n_neighbors` may lead to noisy predictions, while a large `n_neighbors` might oversmooth the decision boundaries or regressions. Additionally, KNN is a non-parametric algorithm and can be computationally expensive for large datasets.

Future steps for improvement could involve exploring other machine learning algorithms, optimizing hyperparameters, and conducting feature engineering to further enhance the model's performance. Additionally, analyzing the importance of different features and their impact on the model's predictions can provide valuable insights for model interpretation and business decision-making.











# McDonalds-Rating-Prediction-Project

This repository contains the code and data for a McDonald's rating prediction project. The goal of the project is to predict the ratings of McDonald's stores based on various features such as store location, review content, and more.


## Introduction

The popularity of McDonald's makes it a significant player in the fast-food industry. Understanding customer feedback and predicting store ratings can provide valuable insights for improving customer satisfaction and business strategies. In this project, we aim to analyze McDonald's customer reviews, visualize the data, and build a predictive model to forecast store ratings.

## Data

The data used in this project is collected from various sources, including customer reviews, store locations, and ratings. The dataset includes features like reviewer ID, store name, category, store address, latitude, longitude, review content, rating count, review time, review, and rating.

## Data Cleaning

Before conducting any analysis or modeling, the data must be cleaned and preprocessed. The data cleaning process involves handling missing values, removing irrelevant or redundant features, and transforming data into a suitable format for analysis. Additionally, we dealt with encoding issues and removed any invalid characters from the review content column.

## Data Visualization

### Histogram of Ratings

A histogram is plotted to visualize the distribution of McDonald's ratings. This helps us understand the overall distribution of ratings given by customers.

![Histogram of Ratings](https://github.com/adodea8991/00-ML/blob/main/Mc-Donalds/Stores-histogram.png)


### Bar Plot of Average Ratings by Store Address

A bar plot is generated to show the average ratings for each store address. This visualization helps us identify any variations in ratings based on different store locations.

![Average rating by store](https://github.com/adodea8991/00-ML/blob/main/Mc-Donalds/Avg-rating-store.png)


### Clustering

In addition to the visualizations mentioned above, we also performed clustering analysis on the dataset. The clustering algorithm groups similar McDonald's restaurants based on certain features such as latitude, longitude, and review-related metrics.

![Store Rating Clustering](https://github.com/adodea8991/00-ML/blob/main/Mc-Donalds/Review-placements.png)


## Modeling

To predict store ratings, we experimented with the Support Vector Regression (SVR) model. SVR is a powerful regression technique that works well for both linear and non-linear relationships between features and target variables. We trained the SVR model on a subset of the data, using features such as latitude, longitude, review length, and review word count to predict store ratings.

## Conclusion

The McDonald's rating prediction project aims to explore customer reviews and store ratings to gain insights into customer satisfaction. We cleaned and visualized the data to better understand the distribution of ratings and the performance of stores across different locations. Finally, we used the Support Vector Regression model to predict store ratings based on relevant features.

Please note that the project is an exploratory analysis and prediction, and further improvements can be made by incorporating additional data and experimenting with different machine learning models.




# Titanic-Data-Analysis-Project

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
python3 missing.py
```

5. The script will read the data from `train.csv`, create a heatmap, and display it. The heatmap will help visualize the missing data in the dataset, making it easier to identify the columns with missing values.


## Decision Trees for Titanic Dataset 

In this repository, we have implemented decision trees for both classification and regression tasks using the Titanic dataset. The Titanic dataset contains various features of passengers, such as age, sex, fare, and class, and whether they survived the disaster or not. The dataset is preprocessed to handle missing values and convert categorical variables to numerical using one-hot encoding.

## Why Decision Trees?

We chose decision trees because they are simple yet powerful algorithms that can be used for both classification and regression tasks. Decision trees are easy to understand and interpret, making them suitable for exploratory data analysis and providing insights into the data.

For classification tasks, decision trees partition the feature space into regions and assign each region the class that is most prevalent within that region. For regression tasks, decision trees assign the average value of the target variable to each leaf node.

## Implementation Details

1. **Preprocessing:** We dropped irrelevant features like "Name," "Ticket," and "Cabin" as they are not expected to have a direct impact on the survival or age of passengers. Rows with missing values were removed from the dataset to ensure data integrity.

2. **One-Hot Encoding:** To handle categorical features like "Sex" and "Embarked," we used one-hot encoding, converting them into numerical representations. This ensures that the decision tree model can handle categorical data effectively.

3. **Decision Tree for Classification:** For the classification task, we used features like "Pclass," "Sex," "Age," "SibSp," "Parch," "Fare," "Embarked_Q," and "Embarked_S" to predict whether a passenger survived or not. The decision tree achieved an accuracy of approximately 0.71 on the test set.

![Visual of the Decision Tree for Classification](https://github.com/adodea8991/00-ML/blob/main/Titanic/Classification-tree.png)


4. **Decision Tree for Regression:** For the regression task, we used features like "Pclass," "Sex," "SibSp," "Parch," "Fare," "Embarked_Q," and "Embarked_S" to predict the age of a passenger. The regression decision tree achieved a mean squared error of approximately 263.37 on the test set.

![Visual of the Decision Tree for Regression](https://github.com/adodea8991/00-ML/blob/main/Titanic/Regression-tree.png)


5. **Visualization:** We visualized both decision trees using the `plot_tree` function from scikit-learn. This provides an intuitive representation of the decision-making process of the tree.

6. **Performance Output:** We displayed the accuracy and mean squared error in a graphical user interface (GUI) using tkinter, providing a quick summary of the model performance.

![Decision Tree Accuracy](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)

7. Impurity Measures Comparison
The visualization of the impurity measures (Gini index and entropy) helps us understand their behavior when splitting data points. The comparison plot shows that the Gini index and entropy are similar in shape, but the Gini index tends to increase slightly more steeply when the proportion of class 1 is close to 0 or 1. This suggests that the Gini index is slightly more sensitive to class imbalance, while entropy provides a more balanced impurity measure regardless of class distribution.

![Impurity Comparison Entropy vs Gini](https://github.com/adodea8991/00-ML/blob/main/Titanic/Impurity-comparison.png)


## Conclusion

The decision tree models achieved reasonably good accuracy and mean squared error on the test data. The classification decision tree demonstrated the capability to predict passenger survival based on given features, while the regression decision tree could predict passenger ages.

Next Steps:
1. **Hyperparameter Tuning:** We can perform hyperparameter tuning to optimize the decision tree models further. Parameters like the maximum depth of the tree and the minimum number of samples required to split a node can be fine-tuned to improve model performance.
2. **Ensemble Methods:** Ensemble methods like Random Forests or Gradient Boosting can be explored to enhance model accuracy and robustness.
3. **Feature Engineering:** Additional feature engineering techniques might be applied to extract more valuable information from the existing features.
4. **Data Scaling:** We can investigate the impact of feature scaling on model performance.
5. **Model Evaluation:** It is essential to evaluate the model's performance on a separate validation set or through cross-validation to ensure it generalizes well to new, unseen data.






# Spam-Filter-Multiple-Algorithms

## Introduction

In this project, we aimed to build a spam filter using three different machine learning algorithms: Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression. The objective of the spam filter is to classify emails as either "spam" or "not spam" based on their content and other relevant features.

## Dataset

We used a dataset named `emails.csv` for training and testing the spam filter. This dataset contains various features extracted from emails, such as word frequencies, subject line, and other attributes. The target variable, "Prediction," indicates whether an email is spam (1) or not spam (0).


![Word Occurance Visualisation](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)


## Methodology

### Data Preprocessing

The first step in building the spam filter involved data preprocessing. We dropped irrelevant columns and performed one-hot encoding on the categorical features, converting them into numerical form for compatibility with the machine learning algorithms.

### Decision Tree


![Decision Tree Accuracy](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)
![Decision Tree Confusion Matrix](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)

The Decision Tree algorithm creates a tree-like model that makes decisions based on feature values. After training the model, we achieved an accuracy of approximately 93%. The confusion matrix indicates that there were 698 true negatives, 41 false positives, 36 false negatives, and 260 true positives.

### K-Nearest Neighbors (KNN)


![KNN Accuracy](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)
![KNN Confusion Matrix](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)

KNN is a classification algorithm that assigns a label to a data point based on the majority class of its k-nearest neighbors. After training the KNN model, we achieved an accuracy of around 86%. The confusion matrix shows 645 true negatives, 94 false positives, 48 false negatives, and 248 true positives.

### Logistic Regression


![Logistic Regression Accuracy](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)
![Logistic Regression Confusion Matrix](https://github.com/adodea8991/00-ML/blob/main/Titanic/Accuracy.png)

Logistic Regression is a linear model used for binary classification. It calculates the probability of a sample belonging to a specific class. The Logistic Regression model yielded the highest accuracy of approximately 97%. The confusion matrix displays 718 true negatives, 21 false positives, 12 false negatives, and 284 true positives.

## Performance Comparison

Overall, each algorithm performed well in classifying spam and non-spam emails. The Logistic Regression model outperformed the other two algorithms in terms of accuracy, achieving a high accuracy of 97%. This suggests that logistic regression was more effective in capturing the underlying patterns in the data.

The confusion matrices provide insights into the model's performance regarding true positives, true negatives, false positives, and false negatives. The Logistic Regression model showed the least number of misclassifications, with fewer false positives and false negatives compared to the Decision Tree and KNN models.

## Conclusion

The spam filter project demonstrated the effectiveness of different machine learning algorithms in identifying spam emails. Logistic Regression emerged as the most accurate classifier among Decision Tree, KNN, and Logistic Regression. However, the choice of the best algorithm depends on the specific requirements of the application and the nature of the data.

To use the spam filter in real-world scenarios, further evaluation and fine-tuning are essential. Additionally, the use of larger and more diverse datasets can enhance the performance and generalizability of the models.










# Housing-Linear-Model

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