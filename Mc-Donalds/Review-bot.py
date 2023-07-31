import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data from the CSV file
data = pd.read_csv("clean_data.csv")

# Preprocess the review text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the value is a string
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        return " ".join(words)
    else:
        return ""

data["processed_review"] = data["review"].apply(preprocess_text)

# Function to search for relevant stores based on user input review term
def search_stores(input_review, data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["processed_review"])
    input_vector = vectorizer.transform([input_review])

    cosine_similarities = linear_kernel(input_vector, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[::-1]
    threshold = 0.2  # Set a threshold for similarity

    relevant_stores = []
    for idx in related_indices:
        if cosine_similarities[idx] > threshold:
            relevant_stores.append(data.iloc[idx])

    return relevant_stores

# User input
user_input_review = input("Tell me what term you'd like me to search for: ")

# Search for relevant stores
relevant_stores = search_stores(preprocess_text(user_input_review), data)

# Output the results
if len(relevant_stores) > 0:
    print("Sure, here are all of the relevant stores:")
    result_df = pd.DataFrame(relevant_stores, columns=["store_address", "review", "rating"])
    print(result_df)
else:
    print("I'm sorry, there seems to be no store with this specific review.")
