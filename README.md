# 🎬 CineSense – AI Movie Review Analyzer

A Streamlit app that classifies an IMDB-style movie review as **Positive 😊** or **Negative 😞**,
built from the notebook `IMDB_Sentiment_Analysis.ipynb`.

## Model & preprocessing
- **Data:** IMDB 50k reviews → 49,582 after removing duplicate reviews; 80/20 stratified split (`random_state=13`).
- **Cleaning:** remove `<br />` tags, lowercase, remove URLs, normalise whitespace.
- **Negation handling (`mark_negation`):** words within 3 tokens after a negator ("not", "never", "n't"…)
  are tagged, e.g. `not very good` → `not NOT_very NOT_good`, until the next punctuation mark.
- **Features:** TF-IDF, 1–3-grams, `sublinear_tf=True`, `min_df=3`, `max_df=0.9`.
- **Classifier:** `LinearSVC(C=1, random_state=42)`.
- The whole thing is one scikit-learn `Pipeline` saved with `joblib`, so raw review text goes straight in.

**Results on the held-out test set (from the notebook, reproduced when the model was re-trained):**

| Model | Accuracy | F1 |
|---|---|---|
| **Negation SVM (final)** | **0.9246** | **0.9254** |
| Stacking | 0.9243 | 0.9251 |
| Linear SVM | 0.9202 | 0.9212 |
| Logistic Regression | 0.9073 | 0.9088 |
| Naive Bayes | 0.8980 | 0.8980 |

LinearSVC does not output calibrated probabilities, so the app shows only the class — no confidence percentage.

## Project structure
```
IMDB_Sentiment_Analysis/
├── app.py                        # Streamlit GUI
├── preprocessing.py              # mark_negation, clean_text, full_preprocess (needed to load the model)
├── sentiment_model.joblib        # fitted pipeline: TF-IDF + LinearSVC
├── train_model.py                # re-creates/saves the model from the IMDB CSV
├── test_app.py                   # automated checks (model + GUI)
├── requirements.txt
├── README.md
└── IMDB_Sentiment_Analysis.ipynb # original notebook
```

## Installation
Python 3.10+ is recommended.
```bash
cd IMDB_Sentiment_Analysis
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```
Then open http://localhost:8501.

## Notes
- The model file was saved with scikit-learn 1.9.1. If you use a different scikit-learn version and see a
  version warning or load error, either install the pinned version or re-train:
  `python train_model.py "path/to/IMDB Dataset.csv"` (takes a few minutes).
- Optional tests: `python test_app.py`
