# train_classical.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
import joblib
import json
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Define models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'naive_bayes': MultinomialNB(),
    'svm': LinearSVC()
}

results = {}
save_dir = "models/classical"
os.makedirs(save_dir, exist_ok=True)

for model_name, clf in models.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        (model_name, clf)
    ])
    pipeline.fit(X_train_processed, y_train)
    predictions = pipeline.predict(X_test_processed)
    results[model_name] = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    pickle_path = os.path.join(save_dir, f'{model_name}.pkl')
    # Save the pipeline
    joblib.dump(pipeline, pickle_path)
    print(f'{model_name} saved with accuracy: {results[model_name]["accuracy"]}')

# Save the metrics for later use by the API
json_path = os.path.join(save_dir, "classical_metrics.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print("Classical models training complete. Metrics saved in classical_metrics.json")
