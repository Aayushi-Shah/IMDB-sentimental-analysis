import os
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from datasets import load_dataset

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load IMDB dataset
imdb_dataset = load_dataset("stanfordnlp/imdb")
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

print("Preprocessing training data...")
X_train_processed = [preprocess_text(text) for text in X_train]
print("Preprocessing test data...")
X_test_processed = [preprocess_text(text) for text in X_test]

# Create a pipeline combining TF-IDF vectorizer and MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Define parameter grid
param_grid = {
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'tfidf__min_df': [3, 5],
    'tfidf__max_df': [0.7, 0.8],
    'nb__alpha': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
print("Starting grid search...")
grid_search.fit(X_train_processed, y_train)

print("Best Parameters:", grid_search.best_params_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

setattr(best_model, "accuracy", accuracy)

# Save the model pipeline to a pickle file
save_dir = "models/classical"
os.makedirs(save_dir, exist_ok=True)
pickle_path = os.path.join(save_dir, "naive_bayes.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Enhanced model saved to {pickle_path}")
