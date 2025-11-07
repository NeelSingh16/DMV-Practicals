# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. Import Dataset
# -------------------------
df = pd.read_csv("./datasets/City_Air_Quality.csv")

# -------------------------
# 2. Explore Dataset
# -------------------------
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

# -------------------------
# 3. Identify Relevant Variables
# -------------------------
# Your dataset columns
aqi_col = "AQI Value"
pollutants = [col for col in ["PM2.5 AQI Value", "PM10 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"] if col in df.columns]

# -------------------------
# 4. Line Plot: AQI Trend by Index
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(df.index, df[aqi_col], label="AQI Value", linewidth=2)
plt.xlabel("Record Index")
plt.ylabel("AQI Value")
plt.title("Overall AQI Trend (Index as Timeline)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# -------------------------
# 5. Line Plots for Pollutants
# -------------------------
plt.figure(figsize=(12, 6))
for pollutant in pollutants:
    plt.plot(df.index, df[pollutant], label=pollutant)
plt.xlabel("Record Index")
plt.ylabel("Pollutant AQI Value")
plt.title("Pollutant Trends Over Records")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# -------------------------
# 6. Bar Plot: Average AQI by City
# -------------------------
city_aqi = df.groupby("City")[aqi_col].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
city_aqi.plot(kind="bar", color="skyblue")
plt.ylabel("Average AQI Value")
plt.title("Top 10 Cities by Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# 7. Box Plot: Distribution of AQI
# -------------------------
plt.figure(figsize=(8, 6))
plt.boxplot(df[aqi_col], vert=True, patch_artist=True)
plt.ylabel("AQI Value")
plt.title("Distribution of AQI Values")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# -------------------------
# 8. Scatter Plot: AQI vs Pollutants
# -------------------------
for pollutant in pollutants:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[pollutant], df[aqi_col], alpha=0.6)
    plt.xlabel(pollutant)
    plt.ylabel("AQI Value")
    plt.title(f"AQI vs {pollutant}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# -------------------------
# 9. Bubble Chart: AQI vs PM2.5 & PM10 (if both available)
# -------------------------
if "PM2.5 AQI Value" in df.columns and "PM10 AQI Value" in df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["PM2.5 AQI Value"],
        df["PM10 AQI Value"],
        s=df[aqi_col]*0.2,
        alpha=0.5,
        c=df[aqi_col],
        cmap="coolwarm"
    )
    plt.xlabel("PM2.5 AQI Value")
    plt.ylabel("PM10 AQI Value")
    plt.title("Bubble Chart: AQI Influence on PM2.5 vs PM10")
    plt.colorbar(label="AQI Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
