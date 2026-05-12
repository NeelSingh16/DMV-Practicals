# ============================================================
# Experiment No. 3 - Classification Analysis using SVM
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: Classify handwritten digits (0-9) using Support Vector Machine
# Dataset: sklearn's built-in digits dataset (same as MNIST-lite)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────
digits = load_digits()
X = digits.data          # Shape: (1797, 64) — 8x8 pixel images flattened
y = digits.target        # Labels: 0 to 9

print("=== Dataset Info ===")
print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}  (8×8 pixel image = 64 values)")
print(f"Classes           : {np.unique(y)}")

# ──────────────────────────────────────────────
# 2. Visualize Sample Images
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    axes[0, digit].imshow(digits.images[idx], cmap='gray_r')
    axes[0, digit].set_title(f"Label: {digit}", fontsize=8)
    axes[0, digit].axis('off')

    idx2 = np.where(y == digit)[0][1]
    axes[1, digit].imshow(digits.images[idx2], cmap='gray_r')
    axes[1, digit].axis('off')

fig.suptitle('Sample Handwritten Digit Images (Rows = Different samples)', fontsize=12)
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150)
plt.show()
print("Sample images saved.")

# ──────────────────────────────────────────────
# 3. Train-Test Split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape}")
print(f"Test  size: {X_test.shape}")

# ──────────────────────────────────────────────
# 4. Feature Scaling
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 5. Train SVM with RBF Kernel
# ──────────────────────────────────────────────
print("\n=== Training SVM with RBF Kernel ===")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train_sc, y_train)

y_pred = svm_model.predict(X_test_sc)

# ──────────────────────────────────────────────
# 6. Evaluate the Model
# ──────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
                             target_names=[str(i) for i in range(10)]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix — SVM (RBF Kernel)  |  Accuracy: {acc*100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

# ──────────────────────────────────────────────
# 7. Compare SVM Kernels
# ──────────────────────────────────────────────
print("\n=== Comparing SVM Kernels ===")
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_scores = {}
for k in kernels:
    clf = SVC(kernel=k, C=10, gamma='scale', random_state=42)
    clf.fit(X_train_sc, y_train)
    score = accuracy_score(y_test, clf.predict(X_test_sc))
    kernel_scores[k] = score
    print(f"  {k:8s}: {score*100:.2f}%")

plt.figure(figsize=(7, 4))
plt.bar(kernel_scores.keys(), [v*100 for v in kernel_scores.values()],
        color=['steelblue', 'coral', 'mediumseagreen', 'violet'])
plt.ylabel('Accuracy (%)')
plt.title('SVM Kernel Comparison')
plt.ylim(50, 105)
for k, v in kernel_scores.items():
    plt.text(k, v*100 + 0.5, f"{v*100:.1f}%", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 8. Cross-Validation
# ──────────────────────────────────────────────
cv_scores = cross_val_score(svm_model, X_train_sc, y_train, cv=5)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ──────────────────────────────────────────────
# 9. Visualize Misclassified Samples
# ──────────────────────────────────────────────
misclassified = np.where(y_pred != y_test)[0]
print(f"\nMisclassified samples: {len(misclassified)} out of {len(y_test)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    if i < len(misclassified):
        idx = misclassified[i]
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray_r')
        ax.set_title(f"True:{y_test[idx]} | Pred:{y_pred[idx]}", fontsize=8, color='red')
        ax.axis('off')
    else:
        ax.axis('off')
fig.suptitle('Misclassified Samples', fontsize=12)
plt.tight_layout()
plt.savefig('misclassified.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# Conclusion
# ──────────────────────────────────────────────
print("\n=== Conclusion ===")
print(f"SVM with RBF kernel achieved {acc*100:.2f}% accuracy on handwritten digit classification.")
print("The kernel trick enables SVM to handle high-dimensional pixel data effectively.")
print(f"Only {len(misclassified)} out of {len(y_test)} test images were misclassified.")
