"""Step 2: Explore the traffic dataset.

Loads data/traffic.csv, prints basic facts about it,
and saves two charts into the results/ folder.
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file into a table (a pandas "DataFrame")
df = pd.read_csv("data/traffic.csv")

# 2. Turn the DateTime text into real dates, so we can sort and group by time
df["DateTime"] = pd.to_datetime(df["DateTime"])

# 3. Print basic facts about the data
print("First 5 rows:")
print(df.head())
print("\nTotal rows:", len(df))
print("\nMissing values in each column:")
print(df.isna().sum())

print("\nRows per junction:")
print(df["Junction"].value_counts().sort_index())

print("\nDate range per junction:")
print(df.groupby("Junction")["DateTime"].agg(["min", "max"]))

print("\nVehicles per hour (summary per junction):")
print(df.groupby("Junction")["Vehicles"].describe().round(1))

# 4. Chart 1: traffic over the whole time period, one line per junction
plt.figure(figsize=(12, 5))
for junction, group in df.groupby("Junction"):
    plt.plot(group["DateTime"], group["Vehicles"], label=f"Junction {junction}", linewidth=0.6)
plt.title("Hourly vehicle counts at 4 junctions")
plt.xlabel("Date")
plt.ylabel("Vehicles per hour")
plt.legend()
plt.tight_layout()
plt.savefig("results/traffic_over_time.png", dpi=150)
print("\nSaved results/traffic_over_time.png")

# 5. Chart 2: the average day - average traffic for each hour (0-23)
df["Hour"] = df["DateTime"].dt.hour
hourly_average = df.groupby(["Hour", "Junction"])["Vehicles"].mean().unstack()

hourly_average.plot(figsize=(10, 5), marker="o")
plt.title("Average traffic by hour of day")
plt.xlabel("Hour of day (0 = midnight)")
plt.ylabel("Average vehicles per hour")
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/average_day.png", dpi=150)
print("Saved results/average_day.png")