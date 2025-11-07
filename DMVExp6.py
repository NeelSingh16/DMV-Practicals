# ----------------------------
# Data Aggregation: Retail Sales Performance by Territory
# ----------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1. Import dataset
df = pd.read_csv("./datasets/sales_data_sample 2.csv", encoding="cp1252")

# 2. Explore dataset
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nSummary Statistics:\n", df.describe())

# 3. Identify relevant variables
# Use TERRITORY as Region, SALES as SalesAmount, PRODUCTLINE as Product Category
region_col = "TERRITORY"
sales_col = "SALES"
category_col = "PRODUCTLINE"

# 4. Group sales data by region and calculate total sales
sales_by_region = df.groupby(region_col)[sales_col].sum().reset_index()
print("\nTotal Sales by Region:\n", sales_by_region)

# 5. Bar Plot: Sales distribution by region
plt.figure(figsize=(8,5))
plt.bar(sales_by_region[region_col], sales_by_region[sales_col], edgecolor="black")
plt.title("Total Sales by Region")
plt.xlabel("Region (Territory)")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.show()

# Pie Chart: Sales distribution by region
plt.figure(figsize=(7,7))
plt.pie(sales_by_region[sales_col], labels=sales_by_region[region_col], autopct='%1.1f%%', startangle=140)
plt.title("Sales Distribution by Region")
plt.show()

# 6. Identify top-performing regions
top_regions = sales_by_region.sort_values(by=sales_col, ascending=False)
print("\nTop Performing Regions:\n", top_regions)

# 7. Group by Region + Product Category
region_category_sales = df.groupby([region_col, category_col])[sales_col].sum().unstack().fillna(0)
print("\nSales by Region and Product Category:\n", region_category_sales)

# 8. Stacked Bar Plot
region_category_sales.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Sales by Region and Product Category (Stacked)")
plt.xlabel("Region (Territory)")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.legend(title="Product Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Grouped Bar Plot
region_category_sales.plot(kind="bar", stacked=False, figsize=(10,6))
plt.title("Sales by Region and Product Category (Grouped)")
plt.xlabel("Region (Territory)")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.legend(title="Product Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
