"""Text helpers shared by the training notebook and the Streamlit app."""
import re

import numpy as np
import pandas as pd

LABELS = ["Negative", "Neutral", "Positive"]


def rating_to_label(r: int) -> str:
    return "Negative" if r <= 2 else ("Neutral" if r == 3 else "Positive")


def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"https?://\S+", " ", s)
    s = s.replace("&#39;", "'").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def combine(title, text) -> str:
    title, text = clean_text(title), clean_text(text)
    return f"{title}. {text}" if title else text


def class_weights(model):
    """Per-class word weights of the final classifier (rows follow model.classes_)."""
    clf = model[-1]
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_)
    if hasattr(clf, "calibrated_classifiers_"):  # CalibratedClassifierCV(LinearSVC)
        return np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
    return np.asarray(clf.feature_log_prob_)  # Naive Bayes


def explain(model, text: str, label: str, top: int = 8) -> pd.DataFrame:
    """Words/phrases in `text` that pushed the prediction towards `label` (or away from it)."""
    vec = model[0]
    x = vec.transform([text])
    W = class_weights(model)
    k = list(model.classes_).index(label)
    others = np.delete(W, k, axis=0).mean(axis=0)
    idx = x.indices
    contrib = x.data * (W[k, idx] - others[idx])
    terms = vec.get_feature_names_out()[idx]
    df = pd.DataFrame({"term": terms, "contribution": contrib})
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top)


def top_terms(model, label: str, n: int = 15) -> list[str]:
    W = class_weights(model)
    k = list(model.classes_).index(label)
    score = W[k] - np.delete(W, k, axis=0).mean(axis=0)
    return list(model[0].get_feature_names_out()[np.argsort(score)[::-1][:n]])
