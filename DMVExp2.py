import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# 1. Setup API
# -------------------------
API_KEY = os.getenv("MY_API_KEY")  # Replace with your OpenWeatherMap API key
CITY = "Mumbai,IN"  # Example: Mumbai, India
URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

# -------------------------
# 2. Fetch Data from API
# -------------------------
response = requests.get(URL)
data = response.json()
pprint(data)

if response.status_code != 200:
    print("Error fetching data:", data)
else:
    print("Data fetched successfully!")

# -------------------------
# 3. Extract Weather Attributes
# -------------------------
weather_list = []

for entry in data["list"]:
    dt = datetime.datetime.fromtimestamp(entry["dt"])
    temp = entry["main"]["temp"]
    humidity = entry["main"]["humidity"]
    wind_speed = entry["wind"]["speed"]
    precipitation = entry.get("rain", {}).get("3h", 0)  # rain in last 3h if available

    weather_list.append({
        "DateTime": dt,
        "Temperature": temp,
        "Humidity": humidity,
        "WindSpeed": wind_speed,
        "Precipitation": precipitation
    })

# Convert to DataFrame
df = pd.DataFrame(weather_list)

print("\n--- Sample Weather Data ---")
print(df.head())

# -------------------------
# 4. Clean & Preprocess
# -------------------------
# Handle missing values
df.fillna(0, inplace=True)

# Ensure DateTime is datetime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Extract Date for daily aggregation
df["Date"] = df["DateTime"].dt.date

# -------------------------
# 5. Data Modeling / Analysis
# -------------------------
print("\nDescriptive Statistics:\n", df.describe())

daily_summary = df.groupby("Date").agg({
    "Temperature": ["mean", "max", "min"],
    "Humidity": "mean",
    "WindSpeed": "mean",
    "Precipitation": "sum"
})

print("\n--- Daily Weather Summary ---")
print(daily_summary)

# -------------------------
# 6. Visualizations
# -------------------------
plt.figure(figsize=(10, 5))
sns.lineplot(x="DateTime", y="Temperature", data=df, marker="o", label="Temp (°C)")
plt.title("Temperature Trend Over Time")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Date", y="Precipitation", data=df, palette="Blues")
plt.title("Daily Precipitation Levels")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Temperature", y="Humidity", data=df, hue="WindSpeed", size="WindSpeed", palette="coolwarm")
plt.title("Temperature vs Humidity (colored by Wind Speed)")
plt.show()

# -------------------------
# 7. Correlation Heatmap
# -------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(df[["Temperature", "Humidity", "WindSpeed", "Precipitation"]].corr(),
            annot=True, cmap="YlGnBu")
plt.title("Correlation between Weather Attributes")
plt.show()
