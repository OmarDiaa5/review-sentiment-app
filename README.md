# Customer Review Sentiment Analysis

Classifies product reviews as **Positive**, **Neutral** or **Negative** and shows what unhappy
customers complain about. The Streamlit app analyses a single review, with the words behind the
result highlighted, or a whole CSV of reviews, with a sentiment dashboard and a downloadable result.

## Results (held-out test set: 4,528 reviews)
Labels come from star ratings: 1-2 = Negative, 3 = Neutral, 4-5 = Positive. 77% of reviews are
positive, so models are compared on **macro F1**, not accuracy.

| Model | CV macro F1 | Accuracy | Macro precision | Macro recall | Macro F1 | Negative recall |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.620 | 0.846 | 0.670 | 0.619 | 0.638 | 0.568 |
| **Logistic Regression (selected)** | **0.657** | **0.829** | **0.644** | **0.661** | **0.652** | **0.597** |
| Linear SVM | 0.625 | 0.845 | 0.678 | 0.614 | 0.637 | 0.584 |

Positive reviews are recognised very reliably. Most errors involve **Neutral** (3-star) reviews,
which usually mix praise and complaints. Full per-class results and confusion matrices:
[reports/evaluation_report.md](reports/evaluation_report.md).

## Project structure
```
review_sentiment_analysis.ipynb  cleaning, labelling, TF-IDF + 3 models, CV, test, saving
text_utils.py                    text cleaning and word-level explanations (shared with the app)
app.py                           Streamlit app
model/                           best_model.joblib (TF-IDF + classifier) + metrics.json
data/sample_reviews.csv          sample file for trying the batch upload
reports/                         evaluation report and charts
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain, put `Womens Clothing E-Commerce Reviews.csv` (Kaggle
`nicapotato/womens-ecommerce-clothing-reviews`) in `data/` and run the notebook.

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, open share.streamlit.io, choose **New app**, select the repo and set the
main file to `app.py`.

## Limitations
- Neutral reviews are the hardest class, and star ratings are noisy labels.
- English clothing reviews only. Other products or languages (e.g. Arabic) need retraining on the
  client's own reviews.
- Sarcasm and very short reviews ("ok") are often misread.
