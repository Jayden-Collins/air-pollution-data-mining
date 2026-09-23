import pandas as pd
import numpy as np

def assign_monsoon(month):
    # Northeast Monsoon (Wet) Jan - Mar, Nov, Dec
    if month in [1, 2, 3, 11, 12]:
        return "NE"
    # Southwest Monsoon (Dry) June - Sep
    elif month in [6, 7, 8, 9]:
        return "SW"
    else:
        return "Inter_Monsoon"

def calc_sub_index(concentration, breakpoints):
    """
    Calculates the sub-index for the specified pollutant

    breakpoints: list of tuples ((Concentration_low, Concentration_high), (Index_low, Index_high))
    """

    # Return early if values are missing
    if pd.isna(concentration):
        return np.nan

    for (c_low, c_high), (i_low, i_high) in breakpoints:
        if c_low <= concentration <= c_high:
            return ((i_high - i_low) / (c_high - c_low) * (
                concentration - c_low
            )) + i_low

    # If concentration exceeds highest breakpoint, cap at max index of 500
    return 500.0

def compute_api_and_categories(df):
    """
    1. Computes individual pollutant sub-indices
    2. Selects highest sub-index to compute overall API score
    3. Returns API score and its health risk category
    """

    # Breakpoints for PM 2.5 (µg/m³)
    pm25_bp = [
      ((0.0, 12.0), (0, 50)),
      ((12.1, 35.4), (51, 100)),
      ((35.5, 55.4), (101, 150)),
      ((55.5, 150.4), (151, 200)),
      ((150.5, 250.4), (201, 300)),
      ((250.5, 500.0), (301, 500)),
    ]

    # Breakpoints for PM10 (µg/m³)
    pm10_bp = [
        ((0, 54), (0, 50)),
        ((55, 154), (51, 100)),
        ((155, 254), (101, 150)),
        ((255, 354), (151, 200)),
        ((355, 424), (201, 300)),
        ((425, 604), (301, 500)),
    ]

    # Breakpoints for CO (ppm)
    co_bp = [
        ((0.0, 4.4), (0, 50)),
        ((4.5, 9.4), (51, 100)),
        ((9.5, 12.4), (101, 150)),
        ((12.5, 15.4), (151, 200)),
        ((15.5, 30.4), (201, 300)),
        ((30.5, 50.0), (301, 500)),
    ]

    # Calculate individual sub-indices
    df["SubIndex_PM25"] = df["PM 2.5"].apply(
        lambda x: calc_sub_index(x, pm25_bp)
    )

    df["SubIndex_PM10"] = df["PM 10"].apply(
        lambda x: calc_sub_index(x, pm10_bp)
    )

    df["SubIndex_CO"] = df["CO"].apply(
        lambda x: calc_sub_index(x, co_bp)
    )

    # Overall API Score is the Maximum Sub-Index across all pollutants
    sub_index_cols = ["SubIndex_PM25", "SubIndex_PM10", "SubIndex_CO"]
    df["API_Score"] = df[sub_index_cols].max(axis=1)

    # Determines API Score Category
    bins = [-1, 50, 100, 150, 200, 300, float("inf")]
    labels = [
        "Good",
        "Fair",
        "Moderate",
        "Poor",
        "Very Poor",
        "Extremely Poor"
    ]

    df["API_Category"] = pd.cut(
        df["API_Score"],
        bins=bins,
        labels=labels,
        right=True
    )

    return df

def clean_df(df):
    # Load dataset
    df = pd.read_parquet("./data/air_pollution.parquet")

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
    ].reset_index(
        drop=True
    )

    """Feature Engineering
    """

    # Particulate Ratio (PM 2.5/PM 10)
    df_final["PM_Ratio"] = df_final["PM 2.5"]/df_final["PM 10"]

    # API Score
    df_final = compute_api_and_categories(df_final)

    # Extract Month
    df_final["Month"] = df_final["date"].dt.month

    # Monsoon Season
    df_final["Monsoon_Season"] = df_final["Month"].apply(assign_monsoon)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(
        df_final,
        columns=["Monsoon_Season"],
        prefix="Monsoon",
        dtype=int,
        drop_first=False, # Retain all categories
    )

    # 1-month lag PM 2.5
    df_encoded["PM25_lag1"] = df_encoded["PM 2.5"].shift(1)

    df_encoded = df_encoded.dropna().reset_index(
        drop=True
    ) # Drop first row due to NaN lag

    return df_encoded
