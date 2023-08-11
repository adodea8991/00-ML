import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy

# Load and preprocess data
data = pd.read_csv('train.csv')
train_texts = data['title'] + ' ' + data['review']
train_labels = data['rating']

# Map rating values to 0-based indices
label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
train_labels = train_labels.map(label_mapping)

# Split the data
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Load tokenizer and encode data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors='pt')

# Create datasets and data loaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels.tolist()))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels.tolist()))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = cross_entropy

# Training loop
for epoch in range(2):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)[0]
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: average loss = {total_loss / len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)[0]
        predicted_labels = torch.argmax(outputs, dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += len(labels)

print(f"Accuracy: {correct / total}")
