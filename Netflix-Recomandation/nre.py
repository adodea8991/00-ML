import pandas as pd
import random
import spacy

# Load the spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Step 1: Load the Netflix dataset from "netflix_titles.csv"
data = pd.read_csv("netflix_titles.csv")

# Step 2: Preprocess the data
selected_data = data[["title", "description"]].dropna()  # Select relevant columns and remove rows with missing values

# Step 3: Select 5 random descriptions
random.seed(42)
sample_data = selected_data.sample(n=5)

# Step 4: Perform Named Entity Recognition on the descriptions
for idx, (movie_name, description) in enumerate(zip(sample_data["title"], sample_data["description"])):
    doc = nlp(description)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Movie Name {idx + 1}: {movie_name}")
    print(f"Named Entities in Description {idx + 1}: {entities}\n")
