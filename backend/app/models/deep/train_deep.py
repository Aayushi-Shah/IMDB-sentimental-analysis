# train_deep.py
import pandas as pd
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Hyperparameters
max_words = 10000
max_len = 200
embedding_dim = 128
batch_size = 32
epochs = 10

# Load dataset
imdb_dataset = load_dataset("stanfordnlp/imdb")
# Split the data
train_data = imdb_dataset["train"]
test_data = imdb_dataset["test"]
X_train = train_data["text"]
y_train = train_data["label"]
X_test = test_data["text"]
y_test = test_data["label"]

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

X_train_processed = [preprocess_text(text) for text in X_train]
X_test_processed = [preprocess_text(text) for text in X_test]


# Tokenize texts
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train_processed)
X_test_seq = tokenizer.texts_to_sequences(X_test_processed)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

results = {}
save_dir = "models/deep"
os.makedirs(save_dir, exist_ok=True)

# Model 1: RNN with LSTM
model_rnn = Sequential([
    Embedding(max_words, embedding_dim),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rnn.fit(X_train_pad, np.array(y_train), validation_data=(X_test_pad, np.array(y_test)),
              batch_size=batch_size, epochs=epochs, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
score_rnn = model_rnn.evaluate(X_test_pad, np.array(y_test), verbose=0)
results['rnn_lstm'] = {'accuracy': score_rnn[1]}
pickle_path = os.path.join(save_dir, 'rnn_lstm.h5')
model_rnn.save(pickle_path)
print(f'RNN LSTM saved with accuracy: {score_rnn[1]}')

# Model 2: Bidirectional LSTM
model_bilstm = Sequential([
    Embedding(max_words, embedding_dim),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])
model_bilstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_bilstm.fit(X_train_pad, np.array(y_train), validation_data=(X_test_pad, np.array(y_test)),
                 batch_size=batch_size, epochs=epochs, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
score_bilstm = model_bilstm.evaluate(X_test_pad, np.array(y_test), verbose=0)
results['bilstm'] = {'accuracy': score_bilstm[1]}
pickle_path = os.path.join(save_dir, 'bilstm.h5')
model_bilstm.save(pickle_path)
print(f'BiLSTM saved with accuracy: {score_bilstm[1]}')

# Model 3: CNN
model_cnn = Sequential([
    Embedding(max_words, embedding_dim),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.fit(X_train_pad, np.array(y_train), validation_data=(X_test_pad, np.array(y_test)),
              batch_size=batch_size, epochs=epochs, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
score_cnn = model_cnn.evaluate(X_test_pad, np.array(y_test), verbose=0)
results['cnn'] = {'accuracy': score_cnn[1]}
pickle_path = os.path.join(save_dir, 'cnn.h5')
model_cnn.save(pickle_path)
print(f'CNN saved with accuracy: {score_cnn[1]}')



# Save the tokenizer for future preprocessing
pickle_path = os.path.join(save_dir, 'tokenizer.pkl')
with open(pickle_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as tokenizer.pkl")

# Save deep model metrics
json_path = os.path.join(save_dir, 'deep_metrics.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print("Deep learning models training complete. Metrics saved in deep_metrics.json")
