import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def eval_rf_model (features, rf_model, y_dev, predictions):
    # Evaluation
    print("MAE:", mean_absolute_error(y_dev, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_dev, predictions)))
    print("R-squared:", r2_score(y_dev, predictions))

    # Feature Importance Chart
    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": features,
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

    # Annotate bars with numeric percentage values
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

    return

def run_rf_model (df):
    # Extract features and target
    features = df[[
        "CO",
        "O3",
        "NO2",
        "SO2",
        'Monsoon_NE',
	    'Monsoon_SW',
	    'Monsoon_Inter_Monsoon',
        "PM25_lag1"
    ]]
    target = df["PM 2.5"]

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=10,
    )

    # Hyperparameter Tuning using GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 1.0],
    }

    # x_train, x_dev, y_train, y_dev = train_test_split(
    #     features,
    #     target,
    #     test_size=0.2,
    #     random_state=10
    # )

    base_rf = RandomForestRegressor(random_state=10)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=kf,
        scoring="r2",
        n_jobs=-1
    )
    grid_search.fit(features, target)

    best_rf = grid_search.best_estimator_

    print('=== Best Hyperparameters ===')
    print(grid_search.best_params_)

    # Evaludate Refined Model across 5 Folds
    cv_results = cross_validate(
        best_rf,
        features,
        target,
        cv=kf,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
    )

    print('\n=== 5-Fold Cross-Validation Performance ===')
    print(f"Mean R-squared: {np.mean(cv_results['test_r2']):.4f}")
    print(f"Mean MAE: {-np.mean(cv_results['test_mae']):.4f}")
    print(f"Mean RMSE: {-np.mean(cv_results['test_rmse']):.4f}")

    return df
