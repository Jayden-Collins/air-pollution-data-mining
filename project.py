import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

'''Load Dataset
'''

# Load dataset
df = pd.read_parquet("./data/air_pollution.parquet")

'''Data Cleaning
'''

# Convert date values into DateTime objects
df["date"] = pd.to_datetime(df["date"])

# Pivot from long form to wide form
df_pivot = df.pivot(
    index="date",
    columns="pollutant",
    values="concentration"
).reset_index() # Reset index instead of using "date" as the new index

# Temporal Filtering: Drop data before Jan 2018
df_final = df_pivot[
    (df_pivot["date"] >= "2018-01-01")
    & (df_pivot["date"] <= "2022-12-31")
].reset_index(drop=True)

'''Exploratory Data Analysis (EDA)
'''

print("Final dataset shape: ", df_final.shape) # Expected: (60, 7) including "date"

print("\nInfo:")
df_final.info()

print("\nSummary:")
print(df_final.describe(include="all"))

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(
    df_final['PM 2.5'],
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
    df_final[pollutant_cols]
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
    df_final.corr(numeric_only=True),
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
    df_final["date"],
    df_final["PM 2.5"],
    marker="o",
    linestyle="-"
)
plt.title("Trend of PM 2.5 Over time (2018 - 2022)")
plt.xlabel("Date")
plt.ylabel("PM 2.5 Concetration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''Random Forest Regressor
'''

# Extract features and target
target = df_final["PM 2.5"]
features = df_final[
    ["CO", "O3", "NO2", "SO2"]
]

x_train, x_dev, y_train, y_dev = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=10
)

# Create and train model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=10
)
rf_model.fit(x_train, y_train)

# Predict outcomes for develpment set
predictions = rf_model.predict(x_dev)

# Evaluation
print("MAE:", mean_absolute_error(y_dev, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_dev, predictions)))
print("R-squared:", r2_score(y_dev, predictions))

# Feature Importance Chart
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=False
)

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=feature_importance_df,
    x="Importance",
    y="Feature",
    palette="viridis"
)

# Annoate bars with numeric percentage values
for index, value in enumerate(feature_importance_df["Importance"]):
    plt.text(
        value + 0.01,
        index,
        f"{value * 100:.1f}%",
    )

plt.title('Feature Importance for Predicting PM2.5 (Random Forest)', fontsize=13)
plt.xlabel('Relative Importance (Gini / Variance Reduction)', fontsize=11)
plt.ylabel('Pollutant Precursor', fontsize=11)
plt.xlim(0, max(feature_importance_df['Importance']) + 0.1)
plt.tight_layout()
plt.show()

# Exact tabular values for report
print("=== Feature Importance Values ===")
print(feature_importance_df.to_string(index=False))
