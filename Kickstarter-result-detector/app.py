import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def classify_text(text, model):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)

    # Make the prediction
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # Define your class labels
    class_labels = ["Failure", "Success"]

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class]

    return predicted_class_label

def main():
    # Load the trained PyTorch model
    model_config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification(config=model_config)
    model.load_state_dict(torch.load("config.json"))

    print("Welcome to the Startup Idea Predictor!")
    print("Enter 'exit' to quit the chatbot.")

    while True:
        # Get input from the user
        user_input = input("\nEnter your startup idea: ")

        # Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Make the prediction
        predicted_class = classify_text(user_input, model)

        # Show the result
        print(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main()
