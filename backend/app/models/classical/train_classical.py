# train_classical.py

import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from datasets import load_dataset

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Enhanced Data Preprocessing Function ---
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation (customize if you need to preserve negation words)
    text = re.sub(r"[^\w\s]", "", text)
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# --- Load the IMDb Dataset from Hugging Face ---
dataset = load_dataset("imdb")

# Extract training and test splits
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# --- Preprocess the Data ---
# The dataset uses 'text' and 'label' columns.
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# --- Addressing Data Imbalance: Shuffle the Training Data ---
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare training and testing sets
X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
y_test = test_df['label']

# --- Improved Feature Engineering: TF-IDF with Unigrams and Bigrams ---
models = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'naive_bayes': MultinomialNB(),
    'svm': LinearSVC()
}

results = {}
for model_name, clf in models.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        (model_name, clf)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    results[model_name] = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    # Save the trained model pipeline
    joblib.dump(pipeline, f'models/classical/{model_name}.pkl')
    print(f'{model_name} saved with accuracy: {results[model_name]["accuracy"]}')

# Save metrics to a JSON file for later use by the API
with open('models/classical/classical_metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Classical models training complete. Metrics saved in models/classical/classical_metrics.json")
