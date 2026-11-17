"""Streamlit app: customer review sentiment analysis (Positive / Neutral / Negative)."""
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from text_utils import LABELS, clean_text, combine, explain

ROOT = Path(__file__).parent
COLORS = {"Positive": "#2ca02c", "Neutral": "#8c8c8c", "Negative": "#d62728"}
STOP = set("""a an the and or but if to of in on at for with is are was were be been it its this that
these those i im i'm me my we our you your she her he his they them their so very too just
really also as from by have has had do does did am will would can could not no don't dont""".split())
st.set_page_config(page_title="Review Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model" / "best_model.joblib")


@st.cache_data
def load_meta():
    return json.loads((ROOT / "model" / "metrics.json").read_text())


model, meta = load_model(), load_meta()


def top_words(texts, n=15):
    """Most common words and 2-word phrases (stop-words removed)."""
    c = Counter()
    for t in texts:
        words = [w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 2]
        c.update(words)
        c.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return pd.DataFrame(c.most_common(n), columns=["word / phrase", "mentions"])


st.title("💬 Review Sentiment Analysis")
st.write("Classify customer reviews as **Positive**, **Neutral** or **Negative**, and see what "
         "unhappy customers talk about.")
st.caption("Trained on English e-commerce clothing reviews. Predictions may be wrong for sarcasm, "
           "very short reviews or other product types; mixed reviews are often labelled Neutral.")

tab_one, tab_batch = st.tabs(["Analyze one review", "Analyze a file of reviews"])

with tab_one:
    text = st.text_area("Customer review", height=130,
                        placeholder="The dress is beautiful but runs small, I had to return it...")
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter a review first.")
        else:
            t = clean_text(text)
            probs = dict(zip(model.classes_, model.predict_proba([t])[0]))
            label = max(probs, key=probs.get)
            msg = f"**{label}** (confidence {probs[label]:.0%})"
            {"Positive": st.success, "Neutral": st.info, "Negative": st.error}[label](msg)
            st.bar_chart(pd.Series(probs).reindex(LABELS).rename("probability"))
            ex = explain(model, t, label)
            ex = ex[ex["contribution"] > 0]
            if len(ex):
                st.write("Words that led to this result: " + " · ".join(
                    f"<span style='color:{COLORS[label]};font-weight:600'>{w}</span>"
                    for w in ex["term"]), unsafe_allow_html=True)

with tab_batch:
    st.write("Upload a CSV with one review per row. You can choose which column holds the text.")
    sample = (ROOT / "data" / "sample_reviews.csv").read_text(encoding="utf-8")
    st.download_button("Download a sample file", sample, "sample_reviews.csv", "text/csv")
    file = st.file_uploader("Reviews CSV", type="csv")
    if file is not None:
        try:
            df = pd.read_csv(file, encoding_errors="replace")
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])
                     or df[c].dtype == object]
        if not text_cols:
            st.error("No text column found in this file.")
            st.stop()
        guess = next((c for c in text_cols if "review" in c.lower() and "text" in c.lower()),
                     next((c for c in text_cols if "review" in c.lower()), text_cols[0]))
        c1, c2, c3 = st.columns(3)
        col = c1.selectbox("Review text column", text_cols, index=text_cols.index(guess))
        title_col = c2.selectbox("Title column (optional)", ["(none)"] + text_cols,
                                 index=(["(none)"] + text_cols).index("Title") if "Title" in text_cols else 0)
        group_col = c3.selectbox("Group by (optional)", ["(none)"] + list(df.columns),
                                 index=(["(none)"] + list(df.columns)).index("Department Name")
                                 if "Department Name" in df.columns else 0)
        if st.button("Analyze all reviews", type="primary"):
            titles = df[title_col] if title_col != "(none)" else [""] * len(df)
            texts = pd.Series([combine(t, r) for t, r in zip(titles, df[col])], index=df.index)
            valid = texts.str.len() > 0
            if not valid.any():
                st.error("The selected column has no text.")
                st.stop()
            out = df[valid].copy()
            probs = model.predict_proba(texts[valid])
            out.insert(0, "confidence", probs.max(axis=1).round(3))
            out.insert(0, "sentiment", model.classes_[probs.argmax(axis=1)])
            counts = out["sentiment"].value_counts().reindex(LABELS, fill_value=0)

            st.write(f"**{len(out):,} reviews analyzed**" +
                     (f" ({(~valid).sum()} empty rows skipped)" if (~valid).any() else ""))
            m = st.columns(3)
            for i, lab in enumerate(LABELS):
                m[i].metric(lab, f"{counts[lab]:,}", f"{counts[lab] / len(out):.0%}",
                            delta_color="off")
            if group_col != "(none)":
                share = pd.crosstab(out[group_col], out["sentiment"], normalize="index") \
                    .reindex(columns=LABELS, fill_value=0)
                st.write(f"Sentiment share by {group_col}")
                st.bar_chart(share, color=[COLORS[l] for l in LABELS], horizontal=True)
            neg = texts[valid][out["sentiment"] == "Negative"]
            if len(neg):
                st.write("What negative reviews mention most")
                st.dataframe(top_words(neg), hide_index=True, width="stretch")
            st.dataframe(out, hide_index=True, width="stretch")
            st.download_button("Download results (CSV)", out.to_csv(index=False),
                               "reviews_with_sentiment.csv", "text/csv")

with st.expander("Model comparison and evaluation"):
    st.write(f"Selected model: **{meta['selected_model']}**")
    st.caption(f"Selected by {meta['selection_rule']}. Labels: {meta['label_rule']}. Test metrics "
               f"come from {meta['test_size']:,} reviews never seen in training.")
    st.dataframe(pd.DataFrame(meta["results"]), hide_index=True, width="stretch")
    for img in ["model_comparison.png", "confusion_matrices.png"]:
        if (ROOT / "reports" / img).exists():
            st.image(str(ROOT / "reports" / img))
