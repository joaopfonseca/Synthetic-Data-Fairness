import os
from os.path import join, dirname
from itertools import product
import pickle
import pandas as pd
from mlresearch.latex import export_longtable, format_table

RESULTS_PATH = join(dirname(__file__), "results")
ANALYSIS_PATH = join(dirname(__file__), "analysis")


def load_model_search_results(results_path):
    results_files = [
        file
        for file in os.listdir(results_path)
        if file.endswith(".pkl") and file.startswith("param_tuning")
    ]

    all_results = []
    for file in results_files:
        results = pickle.load(open(join(results_path, file), "rb"))
        results = (
            results[["param_est_name", "mean_test_score"]]
            .groupby("param_est_name")
            .apply(
                lambda df: df.loc[df["mean_test_score"].idxmax()],
            )
        )
        results["param_est_name"] = results["param_est_name"].str.replace("PREP|", "")
        results["Dataset"] = file.split("_")[2].split(".")[0]
        all_results.append(results)
    df = (
        pd.concat(all_results)
        .reset_index(drop=True)
        .rename(columns={"mean_test_score": "F1", "param_est_name": "Model_Name"})
    )
    df["Feature"] = "overall"
    return df


def load_virny_results(results_path, drop_constant=True):
    results_files = [
        file
        for file in os.listdir(results_path)
        if file.endswith(".csv") and file.startswith("Metrics")
    ]

    all_results = []
    for file in results_files:
        df = pd.read_csv(join(results_path, file))
        model_name = df.loc[0, "Model_Name"]
        df.drop(
            columns=[
                "Runtime_in_Mins",
                "Virny_Random_State",
                "Model_Params",
                "Model_Name",
            ],
            inplace=True,
        )
        # df["Filename"] = file
        df = df.set_index("Metric").T
        df["Model_Name"] = model_name.replace("PREP|", "")
        df.drop(columns="Sample_Size", inplace=True)
        df["Dataset"] = file.split("_")[1]
        df.reset_index(inplace=True)
        df.set_index(["Dataset", "Model_Name", "index"], inplace=True)
        all_results.append(df)

    df = pd.concat(all_results).reset_index().rename(columns={"index": "Feature"})

    return df


def query_results(df, dataset, feature, metric):
    """
    Feature can be one of "overall", "priv", "dis", "correct", "incorrect"

    Metric can be one of "Std", "Overall_Uncertainty", "Statistical_Bias", "IQR",
    "Mean_Prediction", "Aleatoric_Uncertainty", "Epistemic_Uncertainty",
    "Label_Stability", "Jitter", "TPR", "TNR", "PPV", "FNR", "FPR", "Accuracy", "F1",
    "Selection-Rate"
    """
    mask = (df["Dataset"] == dataset) & (df["Feature"].str.endswith(feature))
    df_res = df[mask.values][["Model_Name", metric]]
    df_res["Generator"] = df_res["Model_Name"].str.split("|").str[0]
    df_res["Classifier"] = df_res["Model_Name"].str.split("|").str[1]
    return df_res.pivot(index="Generator", columns="Classifier", values=metric)


if __name__ == "__main__":
    df_param = load_model_search_results(RESULTS_PATH)
    df_virny = load_virny_results(RESULTS_PATH)

    generator_order = ["NONE", "GC", "CONST_GC", "TVAE", "CONST_TVAE"]
    classifier_order = ["LR", "KNN", "DT", "RF"]
    dataset_names = df_virny["Dataset"].unique()
    metric_names = ["FNR", "FPR", "F1", "Accuracy", "Selection-Rate", "PPV"]

    # Overall F1 performance
    for dataset_name in dataset_names:
        f1_scores = query_results(df_param, dataset_name, "overall", "F1")
        f1_scores = f1_scores.map(lambda el: "{0:.3f}".format(el))
        f1_scores = format_table(
            f1_scores,
            generator_order,
            classifier_order,
        )
        f1_scores.index = f1_scores.index.str.replace("_", r"\_")
        f1_scores.reset_index(inplace=True, names="")

        export_longtable(
            f1_scores,
            join(ANALYSIS_PATH, f"overall_f1_{dataset_name}.tex"),
            caption=f"F1 scores for {dataset_name.replace('_', ' ')}.",
            label=f"tab:f1-{dataset_name}",
            index=False,
        )

    # diparities for FNR FPR F1 ACC SELECTION RATE AND PPV
    # per model class do constrained vs unconstrained
    for dataset_name, metric_name in product(dataset_names, metric_names):
        metric_disp = query_results(
            df_virny, dataset_name, "priv", metric_name
        ) / query_results(df_virny, dataset_name, "dis", metric_name)
        metric_disp = metric_disp.map(lambda el: "{0:.3f}".format(el))
        metric_disp = format_table(
            metric_disp,
            generator_order,
            classifier_order,
        )
        metric_disp.index = metric_disp.index.str.replace("_", r"\_")
        metric_disp.reset_index(inplace=True, names="")
        export_longtable(
            metric_disp,
            join(ANALYSIS_PATH, f"metric_disp_{dataset_name}_{metric_name}.tex"),
            caption=(
                f"Disparities for {metric_name.replace('-', ' ')} on "
                f"{dataset_name.replace('_', ' ').title()}."
            ),
            label=f"tab:disp-{dataset_name}-{metric_name}",
            index=False,
        )
