import os
import pickle
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def logistic_regression_improved():
    dataset = load_dataset("stanfordnlp/imdb")
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    pipeline_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],  # Unigrams and bigrams
        "tfidf__min_df": [3, 5],
        "tfidf__max_df": [0.7, 0.8],
        "clf__C": [0.1, 1.0, 10.0]
    }

    grid_search = GridSearchCV(
        pipeline_model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    print("Starting grid search for best logistic regression parameters...")
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    predictions = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))
    
    # Optionally, store the accuracy in the pipeline for later use (e.g., in your API)
    setattr(best_pipeline, "accuracy", accuracy)

  
    model_path = "models/classical/logistic_regression.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    logistic_regression_improved()
