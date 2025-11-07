# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
# 1. Import Dataset
# -------------------------
df = pd.read_csv("datasets/customer_churn_data.csv")

# -------------------------
# 2. Explore Dataset
# -------------------------
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# -------------------------
# 3. Handle Missing Values
# -------------------------
# Example strategy: Fill numeric with mean, categorical with mode
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# -------------------------
# 4. Remove Duplicates
# -------------------------
df.drop_duplicates(inplace=True)

# -------------------------
# 5. Handle Inconsistencies
# -------------------------
# Example: Standardize 'Yes/No' responses
yes_no_cols = ["Churn", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

# Example: Standardize gender
if "gender" in df.columns:
    df["gender"] = df["gender"].str.strip().str.capitalize()

# -------------------------
# 6. Convert Data Types
# -------------------------
# Convert total charges to numeric (if it's object due to spaces)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Convert categorical columns with Label Encoding
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -------------------------
# 7. Handle Outliers
# -------------------------
# Example: Cap extreme values in numerical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                      np.where(df[col] > upper, upper, df[col]))

# -------------------------
# 8. Feature Engineering
# -------------------------
# Example: Create AverageMonthlyCharges = TotalCharges / tenure
if "tenure" in df.columns and "TotalCharges" in df.columns:
    df["AvgMonthlyCharges"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan))
    df["AvgMonthlyCharges"].fillna(df["MonthlyCharges"], inplace=True)

# Example: Create SeniorCitizenCategory
if "SeniorCitizen" in df.columns:
    df["SeniorCategory"] = df["SeniorCitizen"].map({0: "Not Senior", 1: "Senior"})

# Encode new feature
if "SeniorCategory" in df.columns:
    le = LabelEncoder()
    df["SeniorCategory"] = le.fit_transform(df["SeniorCategory"])

# -------------------------
# 9. Normalize / Scale Data
# -------------------------
scaler = StandardScaler()
scaled_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharges"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# -------------------------
# 10. Train-Test Split
# -------------------------
if "Churn" in df.columns:
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nTraining Set Size:", X_train.shape)
    print("Testing Set Size:", X_test.shape)

# -------------------------
# 11. Export Cleaned Data
# -------------------------
df.to_csv("./Output/Cleaned_Telecom_Customer_Churn.csv", index=False)
print("\n✅ Cleaned dataset saved as 'Cleaned_Telecom_Customer_Churn.csv'")

input("Press Enter to exit and close all plots...")
