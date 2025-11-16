import streamlit as st
import pandas as pd
import joblib
from features import clean_text

# Load model + vectorizer
model = joblib.load("../models/model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

st.title("Comment Categorization & Reply Assistant")

uploaded = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded:
    # Read uploaded CSV
    df = pd.read_csv(uploaded)

    # Clean text
    df["cleaned"] = df["text"].apply(clean_text)

    # Transform text
    X_vec = vectorizer.transform(df["cleaned"])

    # Predict
    df["label"] = model.predict(X_vec)

    st.subheader("Predicted Results")
    st.dataframe(df[["text", "label"]])

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "predicted.csv", "text/csv")

    # Visualization
    st.subheader("Label Distribution")
    st.bar_chart(df["label"].value_counts())

else:
    st.info("Please upload a CSV file to begin.")
