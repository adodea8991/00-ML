import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load data from "netflix_titles.csv"
data = pd.read_csv("netflix_titles.csv")

# Step 2: Preprocess the data
selected_data = data[["description", "rating"]].dropna()  # Select relevant columns and remove rows with missing values

# Step 3: Perform 70-30 split for training and testing
X = selected_data["description"]
y = selected_data["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Transform the description text into numerical features (TF-IDF)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train a Support Vector Classification (SVC) model
model = SVC()
model.fit(X_train_tfidf, y_train)

# Step 6: Predict ratings on the test set
y_pred = model.predict(X_test_tfidf)

# Step 7: Visualize the outcome with a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    
    # Filter out rows and columns with all zeros
    cm = cm[~np.all(cm == 0, axis=1)]
    cm = cm[:, ~np.all(cm == 0, axis=0)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(y_test, y_pred, classes=np.unique(y_test), normalize=True, title='Normalized Confusion Matrix')
plt.show()
