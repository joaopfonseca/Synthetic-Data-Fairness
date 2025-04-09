# Base
import os
from os.path import join, dirname
from copy import deepcopy
from itertools import product
import yaml

# Data
# import numpy as np
import pandas as pd
from mlresearch.utils import set_matplotlib_style
from mlresearch.latex import export_table, format_table
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
RESULTS_PATH = join(dirname(__file__), "results", "sample")
ANALYSIS_PATH = join(dirname(__file__), "analysis", "sample")


def load_model_search_results(
    results_path, reference_metric="mean_test_f1",
    get_splits=False, file_prefix="param_tuning", get_param=None
):
    results_files = [
        file
        for file in os.listdir(results_path)
        if file.endswith(".pkl") and file.startswith(file_prefix)
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

        if get_param is not None:
            results[get_param] = results["params"].apply(
                lambda params: [
                    params[key] for key in params.keys() if key.endswith(get_param)
                ]
            ).apply(lambda param_val: param_val[0] if len(param_val) else None)
            groupcols = ["param_est_name", get_param]
        else:
            groupcols = ["param_est_name"]

        results = (
            results[[*groupcols, *all_metrics]]
            .groupby(groupcols)
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
        results = pd.concat([overall_results, *sensitive_results], axis=0)

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
            plt.close()

            (values.T / values.sum(axis=1)).T.plot.bar(stacked=True)
            plt.savefig(join(ANALYSIS_PATH, f"barchart_{name}_target_{attr}"))
            plt.close()


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
                df.groupby(attrs[i - 1: i + 1]).size().to_frame("value").reset_index()
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


def make_boxplots_results(
    metric, results_path, sensitive_attributes, validate_real=True
):
    df_param = load_model_search_results(results_path, get_splits=True)
    synth_mask = df_param.columns.str.endswith("_synth")
    if validate_real:
        df_param = df_param.loc[:, ~synth_mask]
    else:
        df_param = df_param.loc[:, synth_mask]
    sensitive_attributes = deepcopy(sensitive_attributes)
    dataset_names = df_param["Dataset"].unique()
    for name in dataset_names:
        df = df_param[df_param["Dataset"] == name]

        attrs = df["Feature"].unique()
        attrs = attrs[attrs != "overall"]
        for attr in attrs:
            df_ = (
                df[df["Feature"].isin([attr, "overall"])]
                .set_index("Model_Name")[df.columns[df.columns.str.contains(metric)]]
                .drop(columns=f"mean_{metric}", errors="ignore")
                .T
            )
            # df_ = df_.loc[~df_.index.str.endswith(metric)]
            df_["fold"] = df_.index.map(lambda x: x.split("_")[0])
            df_["group"] = df_.index.map(
                lambda x: x.split("_")[-1]
            )
            df_["group"] = df_["group"].map(
                {metric: "overall", "dis": "disadvantaged", "adv": "advantaged"}
            )
            df_ = df_.melt(["group", "fold"]).rename(columns={"value": metric})

            # Sort results
            df_["sortby"] = df_["Model_Name"].map(lambda x: x.split("_")[-1])
            df_ = df_.sort_values(["sortby", "Model_Name"]).drop(columns="sortby")

            # "Gen", "Clf", "IG",
            df_["IG"] = df_["Model_Name"].apply(
                lambda x: x.startswith("CONST_")
            )
            df_["Gen"] = df_["Model_Name"].apply(
                lambda x: x.split("_")[-1].split("|")[0]
            )
            df_["Clf"] = df_["Model_Name"].apply(lambda x: x.split("|")[-1])
            df_.drop(columns=["Model_Name"], inplace=True)

            df_ = df_[(df_["Clf"] != "CONSTANT")]  # & (df_["Gen"] != "NONE")]
            df_["Clf"] = df_["Gen"] + " + " + df_["Clf"]

            fig, axes = plt.subplots(1, 3, sharey=True)
            plt.subplots_adjust(wspace=0)
            for ax, group_name in zip(axes.flatten(), df_["group"].unique()):
                sns.boxplot(
                    df_[df_["group"] == group_name],
                    y=metric,
                    x="Clf",
                    hue="IG",
                    ax=ax
                )
                ax.set_xlabel(group_name)
                ax.set_xticks(
                    ax.get_xticks(),
                    ax.get_xticklabels(),
                    rotation=45,
                    ha='right'
                )
            # fig.suptitle(name)

            plt.savefig(
                join(ANALYSIS_PATH, f"boxplots_{name}_{metric}_{attr}"),
                bbox_inches="tight",
                transparent=False,
            )
            plt.close()


def make_results_table(
    results_path, sensitive_attributes, metric="f1", validate_real=True,
    file_prefix="param_tuning", get_param=None
):
    sensitive_attributes = deepcopy(sensitive_attributes)
    sensitive_attributes = {k: v[0] for k, v in sensitive_attributes.items()}

    df_param = load_model_search_results(
        results_path, get_splits=False, file_prefix=file_prefix, get_param=get_param
    ).copy()

    groupcols = ["Dataset", "Model_Name"]
    if get_param is not None:
        groupcols.append(get_param)

    df_param.set_index([*groupcols, "Feature"], inplace=True)
    synth_mask = df_param.columns.str.endswith("_synth")
    if validate_real:
        df_param = df_param.loc[:, ~synth_mask]
    else:
        df_param = df_param.loc[:, synth_mask]

    df_param.reset_index(inplace=True)
    df_param = df_param.loc[
        df_param["Feature"].isin(list(sensitive_attributes.values())+["overall"])
    ].drop(columns=["Feature"])
    df_param = df_param.groupby(groupcols).mean().reset_index()
    df_param = df_param.melt(groupcols)
    df_param["IG"] = df_param["Model_Name"].apply(
        lambda x: "Yes" if x.startswith("CONST_") else "No"
    )
    df_param["Gen"] = df_param["Model_Name"].apply(
        lambda x: x.split("_")[-1].split("|")[0]
    )
    df_param["Clf"] = df_param["Model_Name"].apply(lambda x: x.split("|")[-1])
    df_param.drop(columns=["Model_Name"], inplace=True)

    # Get metric
    df_param = df_param[df_param["variable"].str.startswith(metric)]
    df_param["variable"] = df_param["variable"].apply(
        lambda x: "ovr" if x == metric else x.split("_")[-1]
    )
    index_cols = ["Gen", "Clf", "IG"]
    if get_param is not None:
        index_cols.append(get_param)
    df_param = (
        df_param
        .sort_values(["Dataset", "Gen", "Clf", "IG", "variable"])
        .pivot(
            columns=["Dataset", "variable"], index=index_cols, values="value"
        )
    )
    return df_param


def get_performance_differences_plot(
    results_path, sensitive_attributes, metric="f1", validate_real=True
):
    df_param = make_results_table(
        results_path, sensitive_attributes, metric, validate_real
    )

    # No IG - With IG
    def _constraints_difference(_df):
        _df["IG"] = 0
        if _df.shape[0] == 2:
            return _df.iloc[0] - _df.iloc[1]
        else:
            return _df.iloc[0]

    df_diff_all = (
        df_param
        .reset_index()
        .groupby(["Gen", "Clf"])
        .apply(_constraints_difference)
        .drop(columns="IG")
    )
    for dataset_name in dataset_names:
        df_diff = df_diff_all.copy()[dataset_name].reset_index()
        df_diff = df_diff[(df_diff["Gen"] != "NONE") & (df_diff["Clf"] != "CONSTANT")]
        df_diff.rename(
            columns={"adv": "advantaged", "dis": "disadvantaged", "ovr": "overall"},
            inplace=True
        )

        fig, axes = plt.subplots(1, 2, sharey=True)
        plt.subplots_adjust(wspace=0)
        for ax, gen_name in zip(axes.flatten(), df_diff["Gen"].unique()):
            df_diff[df_diff["Gen"] == gen_name].set_index("Clf").plot.bar(ax=ax)
            ax.set_xlabel(gen_name)
        # fig.suptitle(dataset_name)

        plt.savefig(
            join(ANALYSIS_PATH, f"perfdiff_{dataset_name}_{metric}"),
            bbox_inches="tight",
            transparent=False,
        )
        plt.close()


def get_param_tuning_analysis(
    results_path, sensitive_attributes, metric="f1", generator="all"
):
    df_results = make_results_table(
        results_path,
        sensitive_attributes,
        metric=metric,
        file_prefix="synth_size",
        get_param="n_rows"
    )
    df_results = df_results.xs('ovr', axis=1, level=1, drop_level=True)
    for dataset_name in df_results.columns:
        df = df_results[dataset_name].to_frame().reset_index()
        df = df[df["Clf"] != "CONSTANT"]

        if generator == "all":
            hue_var = "Gen+Clf"
            df[hue_var] = df["Gen"] + "+" + df["Clf"]
        else:
            hue_var = "Clf"
            df = df[df["Gen"] == generator]

        sns.lineplot(df, x="n_rows", y=dataset_name, hue=hue_var, style="IG")
        plt.ylabel(metric.upper())
        plt.xlabel(r"# Tuples (% of original dataset)")
        plt.savefig(
            join(ANALYSIS_PATH, f"synthsize_line_{dataset_name}_{metric}_{generator}"),
            bbox_inches="tight",
            transparent=False,
        )
        plt.close()


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
    export_table(
        summary.reset_index(),
        join(ANALYSIS_PATH, "datasets_summary.tex"),
        caption="Summary of all datasets used in this study.",
        label="tab:datasets-summary",
        index=False,
        longtable=False
    )
    summary.to_csv(join(ANALYSIS_PATH, "datasets_summary.csv"))

    # EDA: Histogram with % positive / negative per class
    make_target_barchart(datasets, sensitive_attrs)
    sankeys = make_target_sankey_charts(datasets, sensitive_attrs)
    for name, fig in sankeys.items():
        fig.write_image(join(ANALYSIS_PATH, f"sankey_{name}.png"))

    for metric in metric_names:
        make_boxplots_results(metric, RESULTS_PATH, sensitive_attrs)
        get_performance_differences_plot(
            RESULTS_PATH, sensitive_attrs, metric=metric, validate_real=True
        )
        plt.close()

    # Overall metric performance
    for metric in metric_names:
        for dataset_name in dataset_names:
            metric_scores = query_results(df_param, dataset_name, "overall", metric)
            metric_scores = metric_scores.map(lambda el: "{0:.3f}".format(el))
            metric_scores = format_table(
                metric_scores,
                generator_order,
                classifier_order,
            )
            metric_scores.index = metric_scores.index.str.replace("_", r"\_")
            metric_scores.reset_index(inplace=True, names="")

            export_table(
                metric_scores,
                join(ANALYSIS_PATH, f"overall_{metric}_{dataset_name}.tex"),
                caption=f"F1 scores for {dataset_name.replace('_', ' ')}.",
                label=f"tab:f1-{dataset_name}",
                index=False,
                longtable=False
            )
            metric_scores.to_csv(
                join(ANALYSIS_PATH, f"overall_{metric}_{dataset_name}.csv")
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
            export_table(
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
                longtable=False
            )
            metric_disp.to_csv(
                join(
                    ANALYSIS_PATH,
                    f"metric_disp_{dataset_name}_{feature}_{metric_name}.csv",
                )
            )

    # Impact of Synthetic Data Size on performance
    for metric in metric_names:
        get_param_tuning_analysis(
            RESULTS_PATH, sensitive_attrs, metric=metric, generator="TVAE"
        )
