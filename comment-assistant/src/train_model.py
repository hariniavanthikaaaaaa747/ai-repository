import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from features import clean_text

def train_model():
    # Load dataset
    df = pd.read_csv("../prepared.csv")

    # Clean text
    df["cleaned"] = df["text"].apply(clean_text)

    # Features and labels
    X = df["cleaned"]
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=300)
    model.fit(X_train_vec, y_train)

    # Save model + vectorizer
    joblib.dump(model, "../models/model.pkl")
    joblib.dump(vectorizer, "../models/vectorizer.pkl")

    print("Model and vectorizer saved successfully!")

if __name__ == "__main__":
    train_model()
