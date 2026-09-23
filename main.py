import sys

# from src.eda import (
#     plot_monsoon_boxplot,
#     plot_pairplot_matrix,
#     plot_scatter_analyses,
# )

from src.models_rf import run_rf_model
from src.preprocessing import clean_df


def main_menu():
  data_path = "./data/air_pollution.parquet"
  df_processed = None

  while True:
    print("\n" + "=" * 50)
    print(" MALAYSIA AIR POLLUTION DATA MINING SYSTEM ")
    print("=" * 50)
    print("1. Run Data Preprocessing & Feature Engineering")
    print("2. Generate EDA Visualizations")
    print("3. Random Forest Regressor (Supervised)")
    print("4. K-Means Clustering (Unsupervised)")
    print("5. Run Full Data Mining Process")
    print("6. Exit")
    print("=" * 50)

    choice = input("Enter option (1-6): ").strip()

    if choice == "1":
      df_processed = clean_df(data_path)
      print(
          f"Data Preprocessing Complete. Final Shape: {df_processed.shape}"
      )

    elif choice == "2":
      if df_processed is None:
        df_processed = clean_df(data_path)

      print("Generating EDA Visualizations...")

      # plot_scatter_analyses(df_processed)
      # plot_monsoon_boxplot(df_processed)
      # plot_pairplot_matrix(df_processed)

    elif choice == "3":
      if df_processed is None:
        df_processed = clean_df(data_path)

      print("Training Random Forest Regressor...")

      run_rf_model(df_processed)

    elif choice == "4":
      if df_processed is None:
        df_processed = clean_df(data_path)

      print("Executing K-Means Clustering...")

      # df_processed, _ = run_kmeans_clustering(df_processed)

    elif choice == "5":
      print("Running Full Data Mining Pipeline...")
      df_processed = clean_df(data_path)

      # plot_scatter_analyses(df_processed)

      # df_processed, _ = run_kmeans_clustering(df_processed)
      
      run_rf_model(df_processed)

    elif choice == "6":
      print("Exiting program.")
      sys.exit()

    else:
      print("Invalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
  main_menu()
