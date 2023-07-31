import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load the data from the CSV file
df = pd.read_csv('df_text_eng.csv')

# Assuming your text data is in the 'blurb' column and the labels are in the 'state' column.
texts = df['blurb'].tolist()
labels = df['state'].tolist()

# Data Preprocessing
max_sequence_length = 100  # Set the maximum sequence length for padding

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the text data
encoded_data = tokenizer(texts, padding='max_length', truncation=True, return_tensors='tf', max_length=max_sequence_length)

# Convert labels to numerical values (1 for 'successful', 0 for 'failed')
label_map = {'successful': 1, 'failed': 0}
labels = [label_map[label] for label in labels]
labels = np.array(labels)

# Load the DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Training
batch_size = 32
epochs = 10

# Convert the input data and attention masks to NumPy arrays
input_ids = encoded_data['input_ids'].numpy()
attention_mask = encoded_data['attention_mask'].numpy()

model.fit({'input_ids': input_ids, 'attention_mask': attention_mask}, labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Save the trained model
model.save_pretrained('distilbert_model')

# Evaluation (Assuming you have a separate validation dataset)
# Replace 'val_texts' and 'val_labels' with your validation data if available.
# Otherwise, you can use the validation split during training for evaluation.

# val_encoded_data = tokenizer(val_texts, padding='max_length', truncation=True, return_tensors='tf', max_length=max_sequence_length)
# val_input_ids = val_encoded_data['input_ids'].numpy()
# val_attention_mask = val_encoded_data['attention_mask'].numpy()
# val_labels = np.array(val_labels)

# val_loss, val_accuracy = model.evaluate({'input_ids': val_input_ids, 'attention_mask': val_attention_mask}, val_labels, batch_size=batch_size)

# print("Validation Loss:", val_loss)
# print("Validation Accuracy:", val_accuracy)
