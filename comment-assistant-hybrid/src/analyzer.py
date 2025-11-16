import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------
# 1. Initial Setup
# ---------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

# Keyword dictionaries for rule-based emotional/intent detection
praise_words = ["amazing", "great", "love", "awesome", "nice", "good job", "well done"]
support_words = ["keep going", "proud", "stay strong", "we support", "with you"]
criticism_words = ["should improve", "not good", "didn't like", "could be better"]
hate_words = ["hate", "trash", "shut up", "stupid", "idiot"]
threat_words = ["kill", "report you", "block you", "i swear", "watch out"]
emotional_words = ["crying", "tears", "sad", "heartbroken", "reminded me"]
spam_words = ["follow me", "dm me", "promo", "subscribe", "free", "click here"]
question_markers = ["?", "can you", "how to", "what about", "could you"]

# ---------------------------------------------------
# 2. Helper: Check POS structure (grammar tone)
# ---------------------------------------------------
def grammar_analysis(comment):
    tokens = nltk.word_tokenize(comment)
    tags = nltk.pos_tag(tokens)

    # Defensive tone → lots of ! or ALL CAPS
    if comment.isupper() or "!" in comment:
        return "aggressive"

    # If sentence contains many nouns → descriptive/emotional
    noun_count = sum(1 for word, tag in tags if tag.startswith("NN"))
    verb_count = sum(1 for word, tag in tags if tag.startswith("VB"))

    if noun_count > verb_count + 2:
        return "emotional"

    return "neutral"

# ---------------------------------------------------
# 3. Classification Logic
# ---------------------------------------------------
def classify_comment(comment):
    c = comment.lower()
    sentiment = analyzer.polarity_scores(comment)
    tone = grammar_analysis(comment)

    # 1. Check spam
    if any(w in c for w in spam_words):
        return "Spam"

    # 2. Threat
    if any(w in c for w in threat_words):
        return "Threat"

    # 3. Hate/Abuse
    if any(w in c for w in hate_words):
        return "Hate"

    # 4. Praise
    if sentiment["pos"] > 0.5 or any(w in c for w in praise_words):
        return "Praise"

    # 5. Support
    if any(w in c for w in support_words):
        return "Support"

    # 6. Constructive Criticism
    if any(w in c for w in criticism_words) or ("but" in c and sentiment["neg"] > 0.2):
        return "Constructive Criticism"

    # 7. Emotional
    if tone == "emotional" or any(w in c for w in emotional_words):
        return "Emotional"

    # 8. Question
    if any(w in c for w in question_markers):
        return "Question"

    return "Neutral / Other"

# ---------------------------------------------------
# 4. Analyzer Runner
# ---------------------------------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a comment (or 'exit'): ")

        if user_input.lower() == "exit":
            break

        label = classify_comment(user_input)
        print(f"\n🟦 Category: {label}")
