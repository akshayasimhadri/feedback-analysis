!pip install pandas numpy textblob scikit-learn seaborn matplotlib


# -------------------------------
# Customer Feedback Sentiment Analysis Project
# -------------------------------

# Import required libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# NLP libraries
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
# Example: you can use your own CSV file named 'customer_reviews.csv'
# It should have a column named 'feedback' and optionally 'sentiment' for training

# For demonstration, let’s create a small sample dataset
data = {
    'feedback': [
        'I love this product! It works perfectly.',
        'The delivery was late and packaging was poor.',
        'Amazing service, totally satisfied.',
        'Worst purchase ever! Waste of money.',
        'The quality is okay, not great but acceptable.',
        'Customer support was very helpful.',
        'Terrible experience, will not buy again.',
        'Fast shipping and excellent condition!',
        'Average product, expected more.',
        'I’m very happy with my order.'
    ]
}

df = pd.DataFrame(data)

print("Sample Data:\n", df.head(), "\n")

# -------------------------------
# Step 2: Text Preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # remove numbers
    return text

df['clean_feedback'] = df['feedback'].apply(clean_text)

print("After Cleaning:\n", df[['feedback', 'clean_feedback']].head(), "\n")

# -------------------------------
# Step 3: Sentiment using TextBlob
# -------------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_feedback'].apply(get_sentiment)

print("Sentiment Analysis using TextBlob:\n", df[['feedback', 'sentiment']], "\n")

# -------------------------------
# Step 4: Encode for ML Model
# -------------------------------
# Map sentiment to numeric labels
df['label'] = df['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_feedback'], df['label'], test_size=0.3, random_state=42)

# -------------------------------
# Step 5: Feature Extraction (TF-IDF)
# -------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------------------
# Step 6: Train Machine Learning Model
# -------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
# Get the unique labels present in the test data
unique_labels = y_test.unique()
# Map the labels to their corresponding names
label_names = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
target_names_present = [label_names[label] for label in unique_labels]


print("\nClassification Report:\n", classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names_present))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

2300032753_ Akshaya Simhadri, [02-11-2025 16:32]
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg','Pos','Neu'], yticklabels=['Neg','Pos','Neu'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Step 8: Visualize Sentiment Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Step 9: Predict New Feedback
# -------------------------------
def predict_new_feedback(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    pred_label = model.predict(vector)[0]
    sentiment = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}[pred_label]
    return sentiment

sample_feedback = [
    "The service was horrible.",
    "Excellent product! Highly recommend.",
    "It was okay, not too bad."
]

print("\n--- New Feedback Predictions ---")
for fb in sample_feedback:
    print(f"Feedback: {fb} → Sentiment: {predict_new_feedback(fb)}")