# ============================================================
# Experiment No. 2 - Regression Analysis (Uber Fare Prediction)
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: Predict Uber ride price using Linear, Ridge, Lasso Regression
# Dataset: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. Load Dataset
# NOTE: Download 'uber.csv' from the Kaggle link and place it locally.
# For demo, we generate a synthetic dataset if file not found.
# ──────────────────────────────────────────────
try:
    df = pd.read_csv('uber.csv', encoding = 'latin1')
    print("Dataset loaded from file.")
except FileNotFoundError:
    print("uber.csv not found — generating synthetic dataset for demonstration.")
    np.random.seed(42)
    n = 1000
    distance_km = np.random.uniform(1, 30, n)
    duration_min = distance_km * 3 + np.random.normal(0, 2, n)
    hour = np.random.randint(0, 24, n)
    fare = 2.5 + 1.8 * distance_km + 0.05 * duration_min + \
           np.where(hour < 6, 3, 0) + np.random.normal(0, 2, n)
    df = pd.DataFrame({
        'distance_km': distance_km,
        'duration_min': duration_min.clip(1),
        'hour': hour,
        'fare_amount': fare.clip(2.5)
    })

print("\n=== Dataset Info ===")
print(df.head())
print("\nShape:", df.shape)
print("\nData Types:\n", df.dtypes)

# ──────────────────────────────────────────────
# 2. Pre-process the Dataset
# ──────────────────────────────────────────────
print("\n=== Step 1: Pre-processing ===")

# Handle datetime if present (Kaggle uber.csv has pickup_datetime)
if 'pickup_datetime' in df.columns:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, errors='coerce')
    df['hour']      = df['pickup_datetime'].dt.hour
    df['day']       = df['pickup_datetime'].dt.day
    df['month']     = df['pickup_datetime'].dt.month
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek

# Compute distance from lat/lon if present
if all(c in df.columns for c in ['pickup_longitude','pickup_latitude',
                                  'dropoff_longitude','dropoff_latitude']):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                   df['dropoff_latitude'], df['dropoff_longitude'])

# Drop unnecessary columns
drop_cols = ['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
             'dropoff_longitude', 'dropoff_latitude']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Keep fare_amount > 0 and reasonable distance
if 'fare_amount' in df.columns:
    df = df[df['fare_amount'] > 0]
if 'distance_km' in df.columns:
    df = df[df['distance_km'] > 0]

# Remove passenger_count anomalies if present
if 'passenger_count' in df.columns:
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

print(f"Cleaned shape: {df.shape}")

# ──────────────────────────────────────────────
# 3. Identify & Remove Outliers (IQR Method)
# ──────────────────────────────────────────────
print("\n=== Step 2: Outlier Detection (IQR) ===")

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(dataframe)
    dataframe = dataframe[(dataframe[column] >= lower) & (dataframe[column] <= upper)]
    print(f"  {column}: removed {before - len(dataframe)} outliers "
          f"(bounds [{lower:.2f}, {upper:.2f}])")
    return dataframe

for col in ['fare_amount', 'distance_km']:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

print(f"Shape after outlier removal: {df.shape}")

# Visualize fare distribution before/after
plt.figure(figsize=(8, 4))
plt.hist(df['fare_amount'], bins=50, color='steelblue', edgecolor='white')
plt.title('Fare Amount Distribution (after outlier removal)')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('exp2_fare_distribution.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 4. Correlation Analysis
# ──────────────────────────────────────────────
print("\n=== Step 3: Correlation Matrix ===")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('exp2_correlation.png', dpi=150)
plt.show()

print("\nCorrelation with fare_amount:")
print(corr['fare_amount'].sort_values(ascending=False))

# ──────────────────────────────────────────────
# 5. Prepare Features & Target
# ──────────────────────────────────────────────
target = 'fare_amount'
features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

X = df[features].values
y = df[target].values

print(f"\nFeatures used: {features}")

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for Ridge and Lasso)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ──────────────────────────────────────────────
# 6. Train Models: Linear, Ridge, Lasso
# ──────────────────────────────────────────────
print("\n=== Step 4: Training Regression Models ===")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression' : Ridge(alpha=1.0),
    'Lasso Regression' : Lasso(alpha=0.1, max_iter=10000)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'predictions': y_pred}
    print(f"\n  {name}:")
    print(f"    R²   = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE  = {mae:.4f}")

# ──────────────────────────────────────────────
# 7. Compare Models
# ──────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k != 'predictions'}
    for name, vals in results.items()
}).T

print("\n=== Model Comparison ===")
print(metrics_df.round(4))

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, metric in zip(axes, ['R2', 'RMSE', 'MAE']):
    ax.bar(metrics_df.index, metrics_df[metric],
           color=['steelblue', 'coral', 'mediumseagreen'])
    ax.set_title(metric)
    ax.set_xticklabels(metrics_df.index, rotation=15, ha='right')
    ax.set_ylabel(metric)
fig.suptitle('Regression Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('exp2_model_comparison.png', dpi=150)
plt.show()

# Actual vs Predicted scatter for best model
best_model_name = metrics_df['R2'].idxmax()
best_preds = results[best_model_name]['predictions']

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_preds, alpha=0.4, color='steelblue', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title(f'Actual vs Predicted — {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('exp2_actual_vs_predicted.png', dpi=150)
plt.show()

print(f"\nBest Model: {best_model_name} with R² = {metrics_df.loc[best_model_name,'R2']:.4f}")

# ──────────────────────────────────────────────
# Conclusion
# ──────────────────────────────────────────────
print("\n=== Conclusion ===")
print("Linear, Ridge, and Lasso regression models were trained and evaluated.")
print("Ridge and Lasso add regularization to reduce overfitting.")
print(f"Best performer: {best_model_name}")
