# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load Data
# -------------------------
# Load sales data from CSV, Excel, and JSON
csv_data = pd.read_csv("./datasets/sales_data_sample 2.csv", encoding="cp1252")
excel_data = pd.read_excel("./datasets/sales_data_sample 2.xlsx")
json_data = pd.read_json("./datasets/sales_data_sample_2.json", lines=True)

print("CSV Data Sample:\n", csv_data.head())
print("\nExcel Data Sample:\n", excel_data.head())
print("\nJSON Data Sample:\n", json_data.head())

# -------------------------
# 2. Explore Data
# -------------------------
print("\n--- Info about CSV Data ---")
print(csv_data.info())
print("\nMissing values per column:\n", csv_data.isnull().sum())

# -------------------------
# 3. Data Cleaning
# -------------------------
# Combine all data into one DataFrame
sales_data = pd.concat([csv_data, excel_data, json_data], ignore_index=True)

# Remove duplicates
sales_data.drop_duplicates(inplace=True)

# Handle missing values (Example: fill missing SALES with mean)
sales_data["SALES"] = sales_data["SALES"].fillna(sales_data["SALES"].mean())

# Drop rows with critical missing values (e.g., ORDERNUMBER)
sales_data.dropna(subset=["ORDERNUMBER"], inplace=True)

# -------------------------
# 4. Convert to Unified Format
# -------------------------
# Convert ORDERDATE to datetime
sales_data["ORDERDATE"] = pd.to_datetime(sales_data["ORDERDATE"])

# -------------------------
# 5. Data Transformation
# -------------------------
# Derive new variable: Total Revenue (Example: PRICEEACH * QUANTITYORDERED)
sales_data["Revenue"] = sales_data["PRICEEACH"] * sales_data["QUANTITYORDERED"]

# Extract Year and Month for trend analysis
sales_data["Year"] = sales_data["ORDERDATE"].dt.year
sales_data["Month"] = sales_data["ORDERDATE"].dt.month

# -------------------------
# 6. Data Analysis
# -------------------------
print("\n--- Descriptive Statistics ---")
print(sales_data.describe())

# Total Revenue
total_revenue = sales_data["Revenue"].sum()
print(f"\nTotal Revenue: {total_revenue}")

# Average Order Value
avg_order_value = sales_data.groupby("ORDERNUMBER")["Revenue"].sum().mean()
print(f"Average Order Value: {avg_order_value}")

# Sales by Product Line
productline_sales = sales_data.groupby("PRODUCTLINE")["Revenue"].sum()
print("\nSales by Product Line:\n", productline_sales)

# -------------------------
# 7. Data Visualization
# -------------------------

# Sales by Product Line - Bar Plot
plt.figure(figsize=(8,5))
sns.barplot(x=productline_sales.index, y=productline_sales.values, palette="viridis")
plt.title("Total Sales by Product Line")
plt.ylabel("Revenue")
plt.xlabel("Product Line")
plt.xticks(rotation=45, ha='right')
plt.show()

# Country Distribution - Pie Chart
country_sales = sales_data.groupby("COUNTRY")["Revenue"].sum()
plt.figure(figsize=(8,8))
plt.pie(country_sales.values, labels=country_sales.index, autopct='%1.1f%%', startangle=90)
plt.title("Revenue Distribution by Country")
plt.show()

# Revenue Distribution - Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x="PRODUCTLINE", y="Revenue", data=sales_data, palette="Set2")
plt.title("Revenue Distribution by Product Line")
plt.xticks(rotation=45, ha='right')
plt.show()

# Monthly Revenue Trend - Line Plot
monthly_sales = sales_data.groupby(["Year","Month"])["Revenue"].sum().reset_index()
plt.figure(figsize=(10,5))
sns.lineplot(x="Month", y="Revenue", hue="Year", data=monthly_sales, marker="o")
plt.title("Monthly Revenue Trend")
plt.show()
