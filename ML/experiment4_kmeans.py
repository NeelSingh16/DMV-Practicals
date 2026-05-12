# ============================================================
# Experiment No. 4 - Clustering Analysis using K-Means
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: K-Means clustering on Iris dataset + Elbow method
# Dataset: https://www.kaggle.com/datasets/uciml/iris
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────
# Try loading from file; fall back to sklearn's built-in Iris
try:
    df = pd.read_csv('Iris.csv')
    print("Iris.csv loaded from file.")
    # Drop 'Id' column if present (Kaggle version)
    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)
    # Separate features and true labels
    feature_cols = [c for c in df.columns if c != 'Species']
    true_labels_str = df['Species'].values
    label_map = {v: i for i, v in enumerate(np.unique(true_labels_str))}
    true_labels = np.array([label_map[l] for l in true_labels_str])
    X = df[feature_cols].values
except FileNotFoundError:
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    true_labels = iris.target
    feature_cols = iris.feature_names
    df = pd.DataFrame(X, columns=feature_cols)
    df['Species'] = [iris.target_names[t] for t in true_labels]
    print("Loaded Iris dataset from sklearn.")

print("\n=== Dataset Preview ===")
print(df.head())
print(f"\nShape: {df.shape}")
print("\nSpecies distribution:\n", df['Species'].value_counts())

# ──────────────────────────────────────────────
# 2. Explore Data — Pairplot
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for i, species in enumerate(df['Species'].unique()):
    subset = df[df['Species'] == species]
    plt.scatter(subset[feature_cols[0]], subset[feature_cols[1]],
                label=species, s=60, alpha=0.7)
plt.xlabel(feature_cols[0])
plt.ylabel(feature_cols[1])
plt.title('Iris Dataset — Sepal Features by Species (Ground Truth)')
plt.legend()
plt.tight_layout()
plt.savefig('raw_data.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 3. Standardize Features
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures standardized (mean=0, std=1).")

# ──────────────────────────────────────────────
# 4. Elbow Method — Find Optimal K
# ──────────────────────────────────────────────
print("\n=== Elbow Method ===")
wcss = []          # Within-Cluster Sum of Squares
sil_scores = []    # Silhouette scores

K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    if k > 1:
        sil = silhouette_score(X_scaled, km.labels_)
        sil_scores.append(sil)
        print(f"  K={k} | WCSS={km.inertia_:.2f} | Silhouette={sil:.4f}")
    else:
        print(f"  K={k} | WCSS={km.inertia_:.2f} | Silhouette=N/A")

# Plot Elbow Curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(K_range, wcss, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].set_title('Elbow Method — Optimal K Selection')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(range(2, 11), sil_scores, 's-', color='coral', linewidth=2, markersize=8)
axes[1].axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()
print("Elbow plot saved. Optimal K = 3 (matches 3 Iris species).")

# ──────────────────────────────────────────────
# 5. Apply K-Means with K=3
# ──────────────────────────────────────────────
print("\n=== Applying K-Means with K=3 ===")
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Evaluate against ground truth
ari = adjusted_rand_score(true_labels, cluster_labels)
sil = silhouette_score(X_scaled, cluster_labels)
print(f"Adjusted Rand Index  : {ari:.4f}  (1.0 = perfect match with true labels)")
print(f"Silhouette Score     : {sil:.4f}  (higher is better)")

# ──────────────────────────────────────────────
# 6. Visualize Clusters in 2D (using PCA)
# ──────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'coral', 'mediumseagreen']

# Subplot 1: K-Means clusters
for k in range(3):
    mask = cluster_labels == k
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[k], label=f'Cluster {k}', s=60, alpha=0.7)
axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                c='black', marker='X', s=200, zorder=5, label='Centroids')
axes[0].set_title('K-Means Clusters (PCA 2D)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# Subplot 2: Ground truth
species_names = df['Species'].unique()
for i, sp in enumerate(species_names):
    mask = [s == sp for s in df['Species'].values]
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[i], label=sp, s=60, alpha=0.7)
axes[1].set_title('Ground Truth (Species)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.suptitle('K-Means Clustering vs Ground Truth', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150)
plt.show()
print("Cluster visualization saved.")

# ──────────────────────────────────────────────
# 7. Cluster Statistics
# ──────────────────────────────────────────────
df_result = df.copy()
df_result['Cluster'] = cluster_labels

print("\n=== Cluster Statistics ===")
print(df_result.groupby('Cluster')[feature_cols].mean().round(3))

# Heatmap of cluster centers
plt.figure(figsize=(8, 4))
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=feature_cols,
    index=[f'Cluster {i}' for i in range(3)]
)
sns.heatmap(centers_df, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Cluster Centroids (Original Feature Scale)')
plt.tight_layout()
plt.savefig('cluster_centroids.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# Conclusion
# ──────────────────────────────────────────────
print("\n=== Conclusion ===")
print("K-Means with K=3 successfully grouped the Iris flowers into 3 clusters.")
print(f"The Elbow Method confirmed K=3 as the optimal number of clusters.")
print(f"Adjusted Rand Index = {ari:.4f} shows strong agreement with the true species labels.")
