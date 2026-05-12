# ============================================================
# Experiment No. 7 - Emotion Detection from Text (NLP)
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: Classify emotions (joy, anger, sadness, etc.) from text
# Dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# Download required NLTK resources
# ──────────────────────────────────────────────
print("Downloading NLTK resources...")
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ──────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────
try:
    # Kaggle dataset format: text;emotion (semicolon separated, no header)
    df = pd.read_csv('emotions.csv',sep=';',names=['text', 'emotion'], encoding='latin1',engine='python',
        on_bad_lines='skip'
    )
    if df.shape[1] == 1:
        # Try semicolon
        df = pd.read_csv('emotions.csv', sep=';', names=['text', 'emotion'])
    print("emotions.csv loaded from file.")
except FileNotFoundError:
    print("Dataset file not found — using built-in sample emotion data.")
    sample_data = {
        'text': [
            # Joy
            "I am so happy and excited today!",
            "This is the best day of my life!",
            "I feel wonderful and blessed.",
            "I am thrilled about the good news.",
            "Everything is going great, I love it!",
            "I feel so joyful and grateful.",
            "What a delightful surprise!",
            "I am overjoyed with the results.",
            # Sadness
            "I feel so sad and lonely today.",
            "Everything seems hopeless and dark.",
            "I can't stop crying, it hurts so much.",
            "I am heartbroken and devastated.",
            "Life feels empty and meaningless.",
            "I miss them so much, it breaks my heart.",
            "I feel down and depressed.",
            "Nothing seems to go right for me.",
            # Anger
            "I am so angry and furious right now!",
            "This is outrageous and unacceptable!",
            "I hate when people do this to me!",
            "I am absolutely livid about this situation.",
            "Stop doing that, it makes me furious!",
            "I can't believe how rude that was!",
            "I am infuriated by their behavior.",
            "This makes my blood boil.",
            # Fear
            "I am terrified and scared of what might happen.",
            "This situation is giving me anxiety.",
            "I am afraid of failing and being judged.",
            "I feel so worried and nervous.",
            "The thought of it fills me with dread.",
            "I am frightened about the future.",
            "This is really scaring me a lot.",
            "I feel panicked and overwhelmed.",
            # Surprise
            "I can't believe this is happening!",
            "Wow, that was completely unexpected!",
            "I am shocked and amazed by this!",
            "This is so surprising and unbelievable!",
            "I never expected this to turn out this way.",
            "What a shocking revelation!",
            "I am astonished by what just happened.",
            "I'm so surprised I don't know what to say.",
        ],
        'emotion': (
            ['joy'] * 8 +
            ['sadness'] * 8 +
            ['anger'] * 8 +
            ['fear'] * 8 +
            ['surprise'] * 8
        )
    }
    df = pd.DataFrame(sample_data)

print("\n=== Dataset Info ===")
print(df.head(10))
print(f"\nShape: {df.shape}")
print("\nEmotion distribution:\n", df['emotion'].value_counts())

# ──────────────────────────────────────────────
# 2. Visualize Class Distribution
# ──────────────────────────────────────────────
plt.figure(figsize=(9, 4))
emotion_counts = df['emotion'].value_counts()
colors_palette = sns.color_palette('Set2', len(emotion_counts))
bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors_palette)
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Emotion Label Distribution')
for bar, val in zip(bars, emotion_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 3. Text Pre-processing
# ──────────────────────────────────────────────
print("\n=== Text Pre-processing ===")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Pipeline:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove punctuation & numbers
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)         # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)               # Remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)                # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

print("Sample pre-processed texts:")
for _, row in df.head(5).iterrows():
    print(f"  Original : {row['text'][:70]}")
    print(f"  Cleaned  : {row['clean_text']}")
    print(f"  Emotion  : {row['emotion']}\n")

# ──────────────────────────────────────────────
# 4. Feature Extraction (TF-IDF)
# ──────────────────────────────────────────────
print("=== TF-IDF Feature Extraction ===")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X = tfidf.fit_transform(df['clean_text'])
y = df['emotion']

print(f"TF-IDF matrix shape: {X.shape}  ({X.shape[1]} features)")
print(f"Top 15 features: {tfidf.get_feature_names_out()[:15].tolist()}")

# ──────────────────────────────────────────────
# 5. Train-Test Split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ──────────────────────────────────────────────
# 6. Train Multiple Models
# ──────────────────────────────────────────────
print("\n=== Training Classifiers ===")

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes'        : MultinomialNB(alpha=0.1),
    'LinearSVC'          : LinearSVC(random_state=42, max_iter=2000),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'accuracy': acc, 'f1': f1,
                     'predictions': y_pred, 'model': clf}
    print(f"\n  {name}:")
    print(f"    Accuracy         : {acc*100:.2f}%")
    print(f"    Weighted F1 Score: {f1:.4f}")

# ──────────────────────────────────────────────
# 7. Detailed Report for Best Model
# ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best_preds = results[best_name]['predictions']

print(f"\n=== Best Model: {best_name} ===")
print(classification_report(y_test, best_preds))

# Confusion Matrix
cm = confusion_matrix(y_test, best_preds, labels=df['emotion'].unique())
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=df['emotion'].unique(),
            yticklabels=df['emotion'].unique())
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Model comparison chart
model_names = list(results.keys())
accuracies  = [results[m]['accuracy']*100 for m in model_names]
f1_scores   = [results[m]['f1'] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
clr = ['steelblue', 'coral', 'mediumseagreen', 'violet']

axes[0].bar(model_names, accuracies, color=clr)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xticklabels(model_names, rotation=20, ha='right')
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)

axes[1].bar(model_names, f1_scores, color=clr)
axes[1].set_ylabel('Weighted F1 Score')
axes[1].set_title('Model F1 Score Comparison')
axes[1].set_xticklabels(model_names, rotation=20, ha='right')
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=9)

plt.suptitle('Emotion Detection — Model Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 8. Top TF-IDF Words per Emotion
# ──────────────────────────────────────────────
print("\n=== Top Keywords per Emotion (TF-IDF) ===")
best_model = results[best_name]['model']

if hasattr(best_model, 'coef_'):
    classes = best_model.classes_
    feature_names_arr = np.array(tfidf.get_feature_names_out())
    top_n = 8

    fig, axes = plt.subplots(1, len(classes), figsize=(18, 4))
    for ax, cls, coefs in zip(axes, classes, best_model.coef_):
        top_idx = np.argsort(coefs)[-top_n:]
        top_words = feature_names_arr[top_idx]
        top_scores = coefs[top_idx]
        ax.barh(top_words, top_scores, color='steelblue')
        ax.set_title(f'"{cls}"', fontsize=11, fontweight='bold')
        ax.set_xlabel('Coefficient')
        print(f"\n  {cls}: {top_words[::-1].tolist()}")
    plt.suptitle(f'Top Keywords by Emotion ({best_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('top_keywords.png', dpi=150)
    plt.show()

# ──────────────────────────────────────────────
# 9. Interactive Prediction Function
# ──────────────────────────────────────────────
def predict_emotion(text, model=None, vectorizer=None):
    """Predict emotion from raw input text."""
    if model is None:
        model = results[best_name]['model']
    if vectorizer is None:
        vectorizer = tfidf

    clean = preprocess_text(text)
    vec   = vectorizer.transform([clean])
    pred  = model.predict(vec)[0]
    return pred

print("\n=== Emotion Prediction on New Sentences ===")
test_sentences = [
    "I am feeling so wonderful and blessed today!",
    "I can't stop crying; everything is going wrong.",
    "I'm really angry about what happened.",
    "This is scary, I can't believe it!",
    "Wow, what an amazing and unexpected surprise!",
    "I feel empty inside and nothing makes me happy.",
]

for sentence in test_sentences:
    emotion = predict_emotion(sentence)
    print(f"  Input   : {sentence}")
    print(f"  Emotion : {emotion.upper()}\n")

# ──────────────────────────────────────────────
# Conclusion
# ──────────────────────────────────────────────
print("=== Conclusion ===")
print(f"Best Model    : {best_name}")
print(f"Best Accuracy : {results[best_name]['accuracy']*100:.2f}%")
print(f"Best F1 Score : {results[best_name]['f1']:.4f}")
print("\nNLP + TF-IDF + classification successfully detects emotions from text.")
print("This system can be extended to social media monitoring, mental health tools, and chatbots.")
