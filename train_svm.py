# train_tfidf_svm.py

import os
import pickle
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def train_tfidf_svm():
    # Load the IMDB dataset from Hugging Face
    dataset = load_dataset("stanfordnlp/imdb")
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    # Create a pipeline with TF-IDF and LinearSVC
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("classifier", LinearSVC())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Optionally, store the accuracy in the pipeline for later use in the API
    setattr(pipeline, "accuracy", accuracy)

    # Save the trained model to disk
    model_path = "models/classical/tfidf_svm.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

if __name__ == "__main__":
    train_tfidf_svm()
