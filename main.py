import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('datasets/sales_data_sample 2.csv', encoding='cp1252')
# df.to_json('datasets/sales_data_sample_2.json', orient='records', lines=True)
# df1 = pd.read_excel('datasets/sales_data_sample 2.xlsx')
# print(df1)

df = pd.read_csv('datasets/customer_churn_data.csv')
print(df)