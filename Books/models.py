import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the cleaned dataset
df = pd.read_csv("clean_data.csv")

# Drop the "Image" column as it's not needed for the ML model
df.drop(columns=["image"], inplace=True)

# One-hot encode the "Name" and "Author" columns
name_encoder = OneHotEncoder(sparse=False)
name_encoded = name_encoder.fit_transform(df[["name"]])
name_df = pd.DataFrame(name_encoded, columns=name_encoder.get_feature_names_out(["name"]))
author_encoder = OneHotEncoder(sparse=False)
author_encoded = author_encoder.fit_transform(df[["author"]])
author_df = pd.DataFrame(author_encoded, columns=author_encoder.get_feature_names_out(["author"]))
df = pd.concat([df, name_df, author_df], axis=1)
df.drop(columns=["name", "author"], inplace=True)

# Split the data into features (X) and target (y)
X = df.drop(columns=["category"])
y = df["category"]

# Split the data into training and testing sets (75-25 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Create a GUI window
root = tk.Tk()
root.title("Logistic Regression Model Results")

# Create labels to display the results
accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}")
accuracy_label.pack()

report_label = tk.Label(root, text="Classification Report:")
report_label.pack()

# Create a text box to display the classification report
report_textbox = tk.Text(root, width=50, height=20)
report_textbox.insert(tk.END, class_report)
report_textbox.pack()

# Run the GUI event loop
root.mainloop()
