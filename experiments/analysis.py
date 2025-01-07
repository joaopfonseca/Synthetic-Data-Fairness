# Base
import os
from os.path import join, dirname
from copy import deepcopy
from itertools import product
import yaml

# Data
import pandas as pd
from mlresearch.utils import set_matplotlib_style
from mlresearch.latex import export_longtable, format_table
from synfair.datasets import SynFairDatasets

# Viz
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns


DATASET_NAMES = [
    "DIABETES",
    "GERMAN CREDIT",
    "LAW SCHOOL",
    "BANK",
    "CARDIO",
    "CREDIT",
    # "TRAVELTIME",
]
RESULTS_PATH = join(dirname(__file__), "results")
ANALYSIS_PATH = join(dirname(__file__), "analysis")


def load_model_search_results(
    results_path, reference_metric="mean_test_f1", get_splits=False
):
    results_files = [
        file
        for file in os.listdir(results_path)
        if file.endswith(".pkl") and file.startswith("param_tuning")
    ]

    prefix_metric = "split" if get_splits else "mean_test_"

    # Iterate through results (per dataset)
    all_results = []
    for file in results_files:

        # Get results and remove unwanted metrics/results (e.g., train set)
        results = pd.read_pickle(join(results_path, file))
        all_metrics = results.columns[
            results.columns.str.startswith(prefix_metric)
            | (results.columns == reference_metric)
        ]

        # exclude train fold metrics
        all_metrics = all_metrics[~all_metrics.str.contains("train")]

        results = (
            results[["param_est_name", *all_metrics]]
            .groupby("param_est_name")
            .apply(
                lambda df: df.loc[df[reference_metric].idxmax()],
            )
        )
        results["param_est_name"] = results["param_est_name"].str.replace("PREP|", "")
        results["Dataset"] = file.split("_")[2].split(".")[0]

        # Remove "mean_test_" label from columns
        results.columns = results.columns.map(lambda x: x.replace(prefix_metric, ""))
        results.columns = results.columns.map(lambda x: x.replace("test_", ""))

        # "Melt" sensitive features metrics
        sensitive_results = []
        sensitive_features = (
            all_metrics[all_metrics.str.endswith("_dis")]
            .map(lambda x: x.split("_")[-2])
            .unique()
        )
        for sensitive_feature in sensitive_features:
            sensitive_metrics = results.columns[
                results.columns.str.contains(sensitive_feature)
            ]
            df_sens_ = results[["Dataset", "param_est_name", *sensitive_metrics]].copy()
            df_sens_.columns = df_sens_.columns.map(
                lambda x: x.replace(f"{sensitive_feature}_", "")
            )
            df_sens_["Feature"] = sensitive_feature
            df_sens_.set_index(["Dataset", "param_est_name"], inplace=True)
            sensitive_results.append(df_sens_)

        # Get overall metrics
        metrics_cols = results.columns.drop(["Dataset", "param_est_name"])
        metrics_cols = metrics_cols[
            ~(metrics_cols.str.endswith("_dis") | metrics_cols.str.endswith("_adv"))
        ]
        overall_results = results[["Dataset", "param_est_name", *metrics_cols]].copy()
        overall_results["Feature"] = "overall"
        overall_results.set_index(["Dataset", "param_est_name"], inplace=True)
        results = pd.concat([overall_results, *sensitive_results])

        all_results.append(results)

    df = (
        pd.concat(all_results)
        .reset_index()
        .rename(
            columns={
                # reference_metric: reference_metric.split("_")[-1].upper(),
                "param_est_name": "Model_Name"
            }
        )
    )
    return df


# def load_virny_results(results_path, drop_constant=True):
#     results_files = [
#         file
#         for file in os.listdir(results_path)
#         if file.endswith(".csv") and file.startswith("Metrics")
#     ]
#
#     all_results = []
#     for file in results_files:
#         df = pd.read_csv(join(results_path, file))
#         model_name = df.loc[0, "Model_Name"]
#         df.drop(
#             columns=[
#                 "Runtime_in_Mins",
#                 "Virny_Random_State",
#                 "Model_Params",
#                 "Model_Name",
#             ],
#             inplace=True,
#         )
#         # df["Filename"] = file
#         df = df.set_index("Metric").T
#         df["Model_Name"] = model_name.replace("PREP|", "")
#         df.drop(columns="Sample_Size", inplace=True)
#         df["Dataset"] = file.split("_")[1]
#         df.reset_index(inplace=True)
#         df.set_index(["Dataset", "Model_Name", "index"], inplace=True)
#         all_results.append(df)
#
#     df = pd.concat(all_results).reset_index().rename(columns={"index": "Feature"})
#
#     return df


def make_datasets_summary(datasets, sensitive_attributes=None):
    summary = datasets.summarize_datasets()
    summary["Imbalance Ratio"] = summary["Imbalance Ratio"].round(2).astype(str)
    summary.rename(
        columns={
            "Features": "Feats.",
            "Observations": "Obs.",
            "Minority Obs.": "Min. Obs.",
            "Majority Obs.": "Maj. Obs.",
            "Imbalance Ratio": "IR",
            "Constraints": "Const.",
        },
        inplace=True,
    )
    summary.drop(columns=["Classes", "Maj. Obs.", "IR"], inplace=True)
    const_col = summary.pop("Const.")
    summary["Const."] = const_col
    summary.sort_values(by="Obs.", inplace=True)
    if sensitive_attributes is not None:
        summary["SA"] = {k: ", ".join(v) for k, v in sensitive_attributes.items()}
    return summary


def make_target_barchart(datasets, sensitive_attributes):
    sensitive_attributes = deepcopy(sensitive_attributes)
    for name, df in datasets:
        df = df.copy()
        if "age" in sensitive_attributes[name]:
            df["age>60"] = df["age"].map(lambda x: "> 60" if x > 60 else "<=60")
            idx = sensitive_attributes[name].index("age")
            sensitive_attributes[name][idx] = "age>60"

        attrs = sensitive_attributes[name]
        for attr in attrs:
            values = pd.crosstab(df["target"], df[attr])
            values.columns = attr.title() + " " + values.columns.astype(str)

            (values / values.sum(axis=0)).T.plot.bar(stacked=True)
            plt.savefig(join(ANALYSIS_PATH, f"barchart_{name}_{attr}_target"))

            (values.T / values.sum(axis=1)).T.plot.bar(stacked=True)
            plt.savefig(join(ANALYSIS_PATH, f"barchart_{name}_target_{attr}"))


def make_target_sankey_charts(datasets, sensitive_attributes):
    sensitive_attributes = deepcopy(sensitive_attributes)
    figs = {}
    for name, df in datasets:
        df = df.copy()
        if "age" in sensitive_attributes[name]:
            df["age>60"] = df["age"].map(lambda x: "> 60" if x > 60 else "<=60")
            idx = sensitive_attributes[name].index("age")
            sensitive_attributes[name][idx] = "age>60"

        attrs = ["target"] + sensitive_attributes[name]
        all_values = []
        for i in range(1, len(attrs)):
            values = (
                df.groupby(attrs[i - 1 : i + 1]).size().to_frame("value").reset_index()
            )
            for col in values.columns[:-1]:
                values[col] = col.title() + ": " + values[col].astype(str)
            values.columns = ["source", "target", "value"]
            all_values.append(values)
        all_values = pd.concat(all_values)
        label = sorted(
            set(all_values["source"].tolist() + all_values["target"].tolist())
        )

        # setup colors
        target_colormaps = ["red", "green"]
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0, vmax=len(label))
        colors = [to_hex(cmap(norm(i))) for i in range(len(label))]
        i = 0
        for l_ in label:
            if l_.startswith("Target"):
                colors[label.index(l_)] = target_colormaps[i]
                i += 1

        source = all_values["source"].map(lambda x: label.index(x)).tolist()
        target = all_values["target"].map(lambda x: label.index(x)).tolist()
        value = all_values["value"]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node={
                        "label": label,
                        "pad": 15,
                        "thickness": 20,
                        "line": {"width": 0},
                        "color": colors,
                    },
                    link={
                        # indices correspond to labels, eg A1, A2, A1, B1, ...
                        "source": source,
                        "target": target,
                        "value": value,
                    },
                )
            ]
        )
        figs[name] = fig

    return figs


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


def make_boxplots_results(metric, results_path, sensitive_attributes):
    df_param = load_model_search_results(results_path, get_splits=True)
    sensitive_attributes = deepcopy(sensitive_attributes)
    dataset_names = df_param["Dataset"].unique()
    for name in dataset_names:
        df = df_param[df_param["Dataset"] == name]
        # metrics_cols = df.columns[df.columns.map(lambda x: x[0].isnumeric())]
        # metrics_cols = metrics_cols.map(lambda x: "_".join(x.split("_")[1:])).unique()
        # metrics = [f"{metric}_adv", f"{metric}_dis"]
        attrs = df["Feature"].unique()
        attrs = attrs[attrs != "overall"]
        for attr in attrs:
            df_ = (
                df[df["Feature"] == attr]
                .set_index("Model_Name")[df.columns[df.columns.str.contains(metric)]]
                .drop(columns=f"mean_{metric}", errors="ignore")
                .T
            )
            df_ = df_.loc[~df_.index.str.endswith(metric)]
            df_["fold"] = df_.index.map(lambda x: x.split("_")[0])
            df_["group"] = df_.index.map(lambda x: x.split("_")[-1])
            df_ = df_.melt(["group", "fold"]).rename(columns={"value": metric})

            fig, ax = plt.subplots(1, 1)
            sns.boxplot(df_, x=metric, y="Model_Name", hue="group", ax=ax)
            plt.savefig(
                join(ANALYSIS_PATH, f"boxplots_{name}_{metric}_{attr}"),
                bbox_inches="tight",
                transparent=False,
            )


if __name__ == "__main__":
    # matplotlib defaults
    set_matplotlib_style(use_latex=False)

    df_param = load_model_search_results(RESULTS_PATH)

    generator_order = ["NONE", "GC", "CONST_GC", "TVAE", "CONST_TVAE"]
    classifier_order = ["LR", "KNN", "DT", "RF"]
    dataset_names = df_param["Dataset"].unique()
    metric_names = ["fnr", "fpr", "f1", "accuracy", "selection_rate", "ppv"]

    # Save datasets summary
    datasets = SynFairDatasets(names=DATASET_NAMES).download()
    sensitive_attrs = {
        k: list(v.keys())
        for k, v in yaml.safe_load(
            open(join(dirname(__file__), "sensitive_attrs.yml"))
        ).items()
    }
    summary = make_datasets_summary(datasets, sensitive_attrs)
    export_longtable(
        summary.reset_index(),
        join(ANALYSIS_PATH, "datasets_summary.tex"),
        caption="Summary of all datasets used in this study.",
        label="tab:datasets-summary",
        index=False,
    )

    # EDA: Histogram with % positive / negative per class
    make_target_barchart(datasets, sensitive_attrs)
    sankeys = make_target_sankey_charts(datasets, sensitive_attrs)
    for name, fig in sankeys.items():
        fig.write_image(join(ANALYSIS_PATH, f"sankey_{name}.png"))

    for metric in metric_names:
        make_boxplots_results(metric, RESULTS_PATH, sensitive_attrs)

    # Overall F1 performance
    for dataset_name in dataset_names:
        f1_scores = query_results(df_param, dataset_name, "overall", "f1")
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
    for dataset_name in dataset_names:
        sensitive_features = [
            feature
            for feature in df_param[df_param["Dataset"] == dataset_name][
                "Feature"
            ].unique()
            if feature != "overall"
        ]
        for feature, metric_name in product(sensitive_features, metric_names):
            metric_disp = query_results(
                df_param, dataset_name, feature, f"{metric_name}_adv"
            ) / query_results(df_param, dataset_name, feature, f"{metric_name}_dis")
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
                join(
                    ANALYSIS_PATH,
                    f"metric_disp_{dataset_name}_{feature}_{metric_name}.tex",
                ),
                caption=(
                    f"Disparities for metric {metric_name.replace('_', ' ')} in "
                    f"feature {feature} for dataset "
                    f"{dataset_name.replace('_', ' ').title()}."
                ),
                label=f"tab:disp-{dataset_name}-{feature}-{metric_name}",
                index=False,
            )
