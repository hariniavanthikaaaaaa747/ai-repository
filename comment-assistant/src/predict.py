import pandas as pd
import joblib
from features import clean_text

def predict_file(input_path, output_path="../predicted.csv"):
    # Load model + vectorizer
    model = joblib.load("../models/model.pkl")
    vectorizer = joblib.load("../models/vectorizer.pkl")

    # Load input data
    df = pd.read_csv(input_path)

    # Clean text
    df["cleaned"] = df["text"].apply(clean_text)

    # Transform text
    X_vec = vectorizer.transform(df["cleaned"])

    # Predict
    df["predicted_label"] = model.predict(X_vec)

    # Save output
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    predict_file("../prepared.csv")
