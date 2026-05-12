# ============================================================
# Experiment No. 5 - Ensemble Learning: Random Forest Classifier
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: Predict car safety using Random Forest
# Dataset: https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────
# Kaggle car evaluation dataset columns (no header in raw CSV):
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
    df = pd.read_csv('car_evaluation.csv', names=col_names, encoding='latin1')
    print("car_evaluation.csv loaded.")
except FileNotFoundError:
    # Inline data — full UCI Car Evaluation dataset (excerpt for demo)
    print("File not found. Using built-in UCI Car Evaluation dataset.")
    from io import StringIO
    # Full dataset available at: https://archive.ics.uci.edu/ml/datasets/car+evaluation
    raw = """vhigh,vhigh,2,2,small,low,unacc
vhigh,vhigh,2,2,small,med,unacc
vhigh,vhigh,2,2,small,high,unacc
vhigh,vhigh,2,2,med,low,unacc
vhigh,vhigh,2,2,med,med,unacc
vhigh,vhigh,2,2,med,high,unacc
vhigh,vhigh,2,2,big,low,unacc
vhigh,vhigh,2,2,big,med,unacc
vhigh,vhigh,2,2,big,high,unacc
vhigh,vhigh,2,4,small,low,unacc
vhigh,vhigh,2,4,small,med,unacc
vhigh,vhigh,2,4,small,high,unacc
low,low,5more,more,big,high,vgood
low,low,4,more,big,med,good
med,med,4,4,big,high,good
low,low,4,4,big,high,vgood
low,med,4,4,big,high,good
med,low,4,4,big,high,good
low,low,4,4,med,high,good
low,low,4,more,med,high,good"""
    df = pd.read_csv(StringIO(raw), names=col_names)
    print("Note: Using a small demo subset. Download the full dataset for best results.")

print("\n=== Dataset Info ===")
print(df.head(10))
print(f"\nShape: {df.shape}")
print("\nClass distribution:\n", df['class'].value_counts())

# ──────────────────────────────────────────────
# 2. Encode Categorical Features
# All features are categorical — use Label Encoding
# ──────────────────────────────────────────────
print("\n=== Encoding Categorical Variables ===")
le = LabelEncoder()
df_encoded = df.copy()

for col in df.columns:
    df_encoded[col] = le.fit_transform(df[col])
    unique_vals = df[col].unique()
    encoded_vals = df_encoded[col].unique()
    mapping = dict(zip(unique_vals, le.transform(unique_vals)))
    print(f"  {col}: {mapping}")

# ──────────────────────────────────────────────
# 3. Features & Target Split
# ──────────────────────────────────────────────
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

feature_names = X.columns.tolist()
class_names = df['class'].unique()

print(f"\nFeatures: {feature_names}")
print(f"Target classes: {df['class'].unique()}")

# ──────────────────────────────────────────────
# 4. Train-Test Split (80:20)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ──────────────────────────────────────────────
# 5. Train Random Forest Classifier
# ──────────────────────────────────────────────
print("\n=== Training Random Forest Classifier ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# ──────────────────────────────────────────────
# 6. Evaluate the Model
# ──────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")

print("\n=== Classification Report ===")
# Map encoded class back to original names for display
class_labels_ordered = sorted(df['class'].unique())
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix — Random Forest  |  Accuracy: {acc*100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 7. Feature Importance
# ──────────────────────────────────────────────
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\n=== Feature Importance Rankings ===")
print(feat_df.to_string(index=False))

# ──────────────────────────────────────────────
# 8. Effect of Number of Trees
# ──────────────────────────────────────────────
print("\n=== Accuracy vs Number of Trees ===")
n_trees_range = [1, 5, 10, 20, 50, 100, 150, 200]
train_acc, test_acc = [], []

for n in n_trees_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, rf.predict(X_test)))
    print(f"  n_trees={n:4d} | Train: {train_acc[-1]*100:.1f}% | Test: {test_acc[-1]*100:.1f}%")

plt.figure(figsize=(9, 5))
plt.plot(n_trees_range, [a*100 for a in train_acc], 'o-', label='Train Accuracy', color='steelblue')
plt.plot(n_trees_range, [a*100 for a in test_acc],  's-', label='Test Accuracy',  color='coral')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy (%)')
plt.title('Random Forest: Accuracy vs Number of Trees')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('n_trees_accuracy.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 9. Cross-Validation
# ──────────────────────────────────────────────
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ──────────────────────────────────────────────
# Conclusion
# ──────────────────────────────────────────────
print("\n=== Conclusion ===")
print(f"Random Forest Classifier achieved {acc*100:.2f}% accuracy.")
print("Key insights:")
print(f"  - Most important feature: {feat_df.iloc[0]['Feature']}")
print(f"  - Least important feature: {feat_df.iloc[-1]['Feature']}")
print("Random Forest outperforms single decision trees by reducing overfitting via bagging.")
