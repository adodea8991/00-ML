import os
import pandas as pd
import tkinter as tk
import PySimpleGUI as sg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def load_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    sentences = [line.split(';')[0].strip() for line in data]
    emotions = [line.split(';')[1].strip() for line in data]
    return sentences, emotions

def preprocess_text(sentences):
    # Implement any required text preprocessing here if needed
    return sentences

# Load data from txt files
train_sentences, train_emotions = load_data_from_txt('train.txt')
test_sentences, test_emotions = load_data_from_txt('test.txt')
val_sentences, val_emotions = load_data_from_txt('val.txt')

# Step 2: Train the Sentiment Analysis Model
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(preprocess_text(train_sentences))
X_test_tfidf = vectorizer.transform(preprocess_text(test_sentences))

# Data Scaling
scaler = StandardScaler(with_mean=False)  # Set with_mean=False for sparse matrices
X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
X_test_tfidf_scaled = scaler.transform(X_test_tfidf)

# Increase max_iter and train the model with 'liblinear' solver
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_tfidf_scaled, train_emotions)

# Step 3: Create a GUI Interface for Real-Time Sentiment Analysis
def predict_sentiment(sentence):
    tfidf_sentence = vectorizer.transform([preprocess_text(sentence)])
    tfidf_sentence_scaled = scaler.transform(tfidf_sentence)
    prediction = model.predict(tfidf_sentence_scaled)[0]
    return prediction

layout = [
    [sg.Text("Enter a sentence for sentiment analysis:")],
    [sg.InputText(key='-INPUT-')],
    [sg.Button('Analyze'), sg.Button('Reset')],
    [sg.Text(size=(40, 1), key='-OUTPUT-')]
]

window = sg.Window('Sentiment Analysis', layout)

while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == 'Analyze':
        input_sentence = values['-INPUT-']
        if input_sentence:
            sentiment = predict_sentiment(input_sentence)
            window['-OUTPUT-'].update(f"Predicted Sentiment: {sentiment}")
        else:
            window['-OUTPUT-'].update("Please enter a sentence.")
    elif event == 'Reset':
        window['-INPUT-'].update('')
        window['-OUTPUT-'].update('')

window.close()
