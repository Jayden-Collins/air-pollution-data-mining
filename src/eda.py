import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

def eda(df):

    print("Final dataset shape: ", df.shape) # Expected: (60, 7) including "date"

    print("\nInfo:")
    df.info()

    print("\nSummary:")
    print(df.describe(include="all"))

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df['PM 2.5'],
        kde=True,
        bins=20
    )
    plt.title("Distribution of PM 2.5 Concentrations")
    plt.xlabel("PM 2.5 (µg/m³)")
    plt.ylabel("Frequency")
    plt.show()

    # Boxplot
    pollutant_cols = ["PM 2.5", "PM 10", "CO", "O3", "NO2", "SO2"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(
        df[pollutant_cols]
    )

    df_scaled = pd.DataFrame(
        scaled_data,
        columns=[pollutant_cols]
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_scaled
    )
    plt.title("Standardized Boxplot of Pollutant Concentrations (Z-Score, 2018 - 2022)")
    plt.xlabel("Pollutant")
    plt.ylabel("Concentration")
    plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)  # Mean baseline
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f", # format
        vmax=1.0,
        vmin=-1.0,
        center=0.0,
    )
    plt.title("Correlation Heatmap of Pollutants")
    plt.show()

    # Line Chart
    plt.figure(figsize=(14, 6))
    plt.plot(
        df["date"],
        df["PM 2.5"],
        marker="o",
        linestyle="-"
    )
    plt.title("Trend of PM 2.5 Over time (2018 - 2022)")
    plt.xlabel("Date")
    plt.ylabel("PM 2.5 Concetration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
