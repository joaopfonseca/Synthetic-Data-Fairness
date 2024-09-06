import os
from os.path import join, dirname
import pandas as pd

RESULTS_PATH = join(dirname(__file__), "results")


def load_results(results_path):
    results_files = [file for file in os.listdir(results_path) if file.endswith(".csv")]

    all_results = []
    for file in results_files:
        df = pd.read_csv(join(results_path, file))
        df["Filename"] = file
        all_results.append(df)

    df = pd.concat(all_results)
    df["Dataset"] = df["Filename"].str.split("_").str[1]
    df.drop(columns=["Filename"], inplace=True)
    return df


if __name__ == "__main__":
    df_res = load_results(RESULTS_PATH)
