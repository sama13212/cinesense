"""
Re-creates the final model from the notebook (Negation SVM, `neg_pipe`) and saves it.

Usage:  python train_model.py path/to/"IMDB Dataset.csv"
Same split as the notebook: test_size=0.2, random_state=13, stratified.
"""
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text, full_preprocess

csv_path = sys.argv[1] if len(sys.argv) > 1 else "IMDB Dataset.csv"
df = pd.read_csv(csv_path)
df.drop_duplicates(subset=["review"], inplace=True)          # notebook cell 13
df["clean_review"] = df["review"].map(clean_text)             # notebook cells 31-37
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

x, y = df["clean_review"], df["sentiment"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=13, stratify=y)

neg_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=full_preprocess,
                              token_pattern=r"[A-Za-z_']+",
                              sublinear_tf=True, ngram_range=(1, 3),
                              min_df=3, max_df=0.9)),
    ("svm", LinearSVC(C=1, random_state=42)),
])
neg_pipe.fit(x_train, y_train)

pred = neg_pipe.predict(x_test)
print("Train accuracy:", neg_pipe.score(x_train, y_train))
print("Test accuracy :", accuracy_score(y_test, pred))
print("Test F1       :", f1_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(neg_pipe, "sentiment_model.joblib", compress=3)
print("Saved sentiment_model.joblib")
