# train_deep.py

import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
import os
from datasets import load_dataset

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Enhanced Data Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Hyperparameters
max_words = 10000
max_len = 200
embedding_dim = 100  # Using 100-dimensional GloVe embeddings
batch_size = 32
epochs = 10  # Increase epochs for better performance

# --- Load the IMDb Dataset from Hugging Face ---
dataset = load_dataset("imdb")

# Extract training and test splits
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# --- Preprocess the Data ---
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# --- Addressing Data Imbalance: Shuffle the Training Data ---
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare training and testing sets
X_train = train_df['text'].tolist()
y_train = train_df['label'].tolist()
X_test = test_df['text'].tolist()
y_test = test_df['label'].tolist()

# --- Tokenize the Texts ---
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

results = {}

embedding_index = {}
glove_path = 'models/deep_learning/glove.6B.100d.txt'
with open(glove_path, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# # --- Model 1: RNN with LSTM ---
# model_rnn = Sequential([
#     Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False),
#     LSTM(64),
#     Dense(1, activation='sigmoid')
# ])
# model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_rnn.fit(X_train_pad, np.array(y_train), validation_data=(X_test_pad, np.array(y_test)),
#               batch_size=batch_size, epochs=epochs,
#               callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
# score_rnn = model_rnn.evaluate(X_test_pad, np.array(y_test), verbose=0)
# results['rnn_lstm'] = {'accuracy': score_rnn[1]}
# model_rnn.save('models/deep_learning/rnn_lstm.h5')
# print(f'RNN LSTM saved with accuracy: {score_rnn[1]}')

# # --- Model 2: Bidirectional LSTM ---
# model_bilstm = Sequential([
#     Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False),
#     Bidirectional(LSTM(64)),
#     Dense(1, activation='sigmoid')
# ])
# model_bilstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_bilstm.fit(X_train_pad, np.array(y_train), validation_data=(X_test_pad, np.array(y_test)),
#                  batch_size=batch_size, epochs=epochs,
#                  callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
# score_bilstm = model_bilstm.evaluate(X_test_pad, np.array(y_test), verbose=0)
# results['bilstm'] = {'accuracy': score_bilstm[1]}
# model_bilstm.save('models/deep_learning/bilstm.h5')
# print(f'BiLSTM saved with accuracy: {score_bilstm[1]}')

# # --- Model 3: CNN ---
# model_cnn = Sequential([
#     Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False),
#     Conv1D(128, 5, activation='relu'),
#     GlobalMaxPooling1D(),
#     Dense(1, activation='sigmoid')
# ])
# model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_cnn.fit(X_train_pad, np.array(y_train), validation_data=(X_test_pad, np.array(y_test)),
#               batch_size=batch_size, epochs=epochs,
#               callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
# score_cnn = model_cnn.evaluate(X_test_pad, np.array(y_test), verbose=0)
# results['cnn'] = {'accuracy': score_cnn[1]}
# model_cnn.save('models/deep_learning/cnn.h5')
# print(f'CNN saved with accuracy: {score_cnn[1]}')

# Save the tokenizer for future preprocessing during inference
with open('models/deep_learning/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as models/deep_learning/tokenizer.pkl")

# Save deep learning model metrics for later API use
with open('models/deep_learning/deep_metrics.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Deep learning models training complete. Metrics saved in models/deep_learning/deep_metrics.json")
