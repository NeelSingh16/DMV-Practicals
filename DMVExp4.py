# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 1. Import Dataset & Clean Column Names
# -------------------------
df = pd.read_csv("./datasets/RealEstate_Prices.csv")

# Clean column names (remove spaces, lowercase, replace special chars with _)
df.columns = df.columns.str.strip().str.lower().str.replace("[^a-z0-9]", "_", regex=True)

print("\n--- First 5 Rows ---")
print(df.head())

# -------------------------
# 2. Handle Missing Values
# -------------------------
print("\nMissing Values per Column:\n", df.isnull().sum())

# Example: Numeric -> fill with mean, Categorical -> fill with mode
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# -------------------------
# 3. Data Merging (Optional if additional dataset exists)
# -------------------------
# Example: Merge neighborhood demographics dataset if available
# demo_df = pd.read_csv("Neighborhood_Demographics.csv")
# df = df.merge(demo_df, on="neighborhood", how="left")

# -------------------------
# 4. Filter & Subset Data
# -------------------------
# Example: Select properties sold after 2015, only Residential in "Downtown"
if "year_sold" in df.columns:
    df = df[df["year_sold"] >= 2015]
if "property_type" in df.columns and "location" in df.columns:
    df = df[(df["property_type"] == "Residential") & (df["location"] == "Downtown")]

print("\nFiltered Dataset Shape:", df.shape)

# -------------------------
# 5. Handle Categorical Variables
# -------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns

# Option A: Label Encoding (good for tree models)
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Option B: One-Hot Encoding (uncomment if needed for regression/ML models)
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------
# 6. Aggregation / Summary Stats
# -------------------------
if "neighborhood" in df.columns:
    avg_price_by_neighborhood = df.groupby("neighborhood")["sale_price"].mean()
    print("\nAverage Sale Price by Neighborhood:\n", avg_price_by_neighborhood)

if "property_type" in df.columns:
    avg_price_by_type = df.groupby("property_type")["sale_price"].mean()
    print("\nAverage Sale Price by Property Type:\n", avg_price_by_type)

# -------------------------
# 7. Handle Outliers
# -------------------------
if "sale_price" in df.columns:
    Q1 = df["sale_price"].quantile(0.25)
    Q3 = df["sale_price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Cap extreme values
    df["sale_price"] = np.where(df["sale_price"] < lower, lower,
                                np.where(df["sale_price"] > upper, upper, df["sale_price"]))

# -------------------------
# Final Dataset Ready
# -------------------------
print("\n✅ Data Wrangling Complete. Final Shape:", df.shape)

# Export cleaned dataset
df.to_csv("./Output/Cleaned_RealEstate_Prices.csv", index=False)
print("\nCleaned dataset saved as 'Cleaned_RealEstate_Prices.csv'")
