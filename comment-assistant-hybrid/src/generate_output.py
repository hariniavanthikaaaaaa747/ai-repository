import pandas as pd
from analyzer import classify_comment, grammar_analysis

# Load your dataset of 200 comments
df = pd.read_csv("../prepared.csv")

emotions = []
structures = []

print("Processing all comments...")

for comment in df['text']:
    # Use analyzer functions
    emotion = classify_comment(comment)
    structure = grammar_analysis(comment)

    emotions.append(emotion)
    structures.append(structure)

df['emotion'] = emotions
df['structure'] = structures

# Save final output
df.to_csv("../hybrid_output.csv", index=False)

print("Done! File saved as hybrid_output.csv")
