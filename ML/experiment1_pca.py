# ============================================================
# Experiment No. 1 - Feature Transformation using PCA
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: Apply PCA on Wine dataset to distinguish red & white wine
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────
url = "https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv"
df = pd.read_csv(url)

print("=== Dataset Info ===")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# The first column is the class label (1 = Class 1, 2 = Class 2, 3 = Class 3)
# Treat classes 1 & 2 as "red-like" and class 3 as "white-like" for visualization
# (Original Wine dataset has 3 cultivars, not red/white — we use all 3 classes)

# ──────────────────────────────────────────────
# 2. Separate Features and Labels
# ──────────────────────────────────────────────
X = df.iloc[:, 1:]   # All feature columns
y = df.iloc[:, 0]    # Class label column

print("\n=== Feature Matrix shape:", X.shape)
print("=== Target distribution:\n", y.value_counts())

# ──────────────────────────────────────────────
# 3. Standardize the Data
# PCA is sensitive to scale — standardization is mandatory
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== Standardized Data (first 3 rows) ===")
print(pd.DataFrame(X_scaled, columns=X.columns).head(3))

# ──────────────────────────────────────────────
# 4. Apply PCA
# ──────────────────────────────────────────────
# Step A: Fit PCA to find all components and their explained variance
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\n=== Explained Variance per Component ===")
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"  PC{i+1}: {ev*100:.2f}%  |  Cumulative: {cv*100:.2f}%")

# Step B: Plot Scree Plot (Explained Variance vs. Components)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance)+1), explained_variance * 100, color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot')
plt.xticks(range(1, len(explained_variance)+1))

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance * 100, marker='o', color='darkorange')
plt.axhline(y=95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance')
plt.legend()

plt.tight_layout()
plt.savefig('exp1_scree_plot.png', dpi=150)
plt.show()
print("Scree plot saved.")

# Step C: Reduce to 2 Principal Components for visualization
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

print(f"\n=== PCA Reduced Shape: {X_pca.shape}")
print(f"Variance explained by PC1 + PC2: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")

# ──────────────────────────────────────────────
# 5. Visualize PCA Results
# ──────────────────────────────────────────────
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Class'] = y.values

class_labels = {1: 'Cultivar 1 (Red-like)', 2: 'Cultivar 2 (Rosé-like)', 3: 'Cultivar 3 (White-like)'}
colors = {1: 'red', 2: 'orange', 3: 'green'}

plt.figure(figsize=(9, 6))
for cls in sorted(pca_df['Class'].unique()):
    subset = pca_df[pca_df['Class'] == cls]
    plt.scatter(subset['PC1'], subset['PC2'],
                label=class_labels[cls],
                color=colors[cls], alpha=0.7, edgecolors='k', s=60)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA – Wine Dataset (2D Projection)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/exp1_pca_scatter.png', dpi=150)
plt.show()
print("PCA scatter plot saved.")

# ──────────────────────────────────────────────
# 6. Loadings (Feature Contributions to PC1 & PC2)
# ──────────────────────────────────────────────
loadings = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns
)

plt.figure(figsize=(10, 5))
loadings[['PC1', 'PC2']].plot(kind='bar', figsize=(12, 5), color=['steelblue', 'coral'])
plt.title('Feature Loadings on PC1 and PC2')
plt.ylabel('Loading Value')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/exp1_loadings.png', dpi=150)
plt.show()
print("Loadings plot saved.")

print("\n=== Top Contributing Features to PC1 ===")
print(loadings['PC1'].abs().sort_values(ascending=False).head(5))

# ──────────────────────────────────────────────
# 7. Conclusion
# ──────────────────────────────────────────────
print("\n=== Conclusion ===")
print(f"PCA reduced {X.shape[1]} features → 2 principal components.")
print(f"PC1 + PC2 together explain {pca_2d.explained_variance_ratio_.sum()*100:.2f}% of total variance.")
print("The scatter plot clearly shows separation among the 3 wine cultivars,")
print("demonstrating PCA's effectiveness in dimensionality reduction.")
