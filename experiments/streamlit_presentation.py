from os.path import join, dirname
import yaml
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SynFair Analysis")

DATASET_NAMES = [
    "SYNTHETIC",
    "DIABETES",
    "GERMAN CREDIT",
    "LAW SCHOOL",
    "BANK",
    "CARDIO",
    "CREDIT",
]
RESULTS_PATH = join(dirname(__file__), "results", "sample")
ANALYSIS_PATH = join(dirname(__file__), "analysis", "sample")

SENSITIVE_ATTRS = {
    k: list(v.keys())
    for k, v in yaml.safe_load(
        open(join(dirname(__file__), "sensitive_attrs.yml"))
    ).items()
}

GENERATORS = ["NONE", "GC", "CONST_GC", "TVAE", "CONST_TVAE"]
CLASSIFIERS = ["LR", "KNN", "DT", "RF"]
METRIC_NAMES = ["fnr", "fpr", "f1", "accuracy", "selection_rate", "ppv"]

########################################################################################
# Sidebar
########################################################################################

dataset_name = st.sidebar.selectbox(
    "Dataset", ["Overall"] + [x.title() for x in DATASET_NAMES]
).upper()

metric = st.sidebar.selectbox(
    "Metric", METRIC_NAMES, 2
)

if (dataset_name != "OVERALL") and dataset_name != "SYNTHETIC":
    attr = st.sidebar.selectbox(
        "Sensitive Attribute", SENSITIVE_ATTRS[dataset_name]
    )

########################################################################################
# Main
########################################################################################


st.write(f"# {dataset_name.title()}")
if (dataset_name != "OVERALL") and dataset_name != "SYNTHETIC":
    # Individual analysis
    tab1, tab2 = st.tabs(["Overview", "Results"])
    with tab1:
        # Overview
        ##########
        st.write(f"Sensitive attributes: {', '.join(SENSITIVE_ATTRS[dataset_name])}")
        st.image(join(ANALYSIS_PATH, f"synthsize_line_{dataset_name}_{metric}_TVAE.png"))
        col1, col2 = st.columns(2)
        with col1:
            attr_ = attr if attr != "age" else "age>60"
            st.image(join(ANALYSIS_PATH, f"barchart_{dataset_name}_{attr_}_target.png"))
        with col2:
            st.image(join(ANALYSIS_PATH, f"barchart_{dataset_name}_target_{attr_}.png"))
        st.image(join(ANALYSIS_PATH, f"sankey_{dataset_name}.png"))
    with tab2:
        # Results
        #########

        # Tables
        col1, col2 = st.columns(2)
        with col1:
            df_res = pd.read_csv(
                join(ANALYSIS_PATH, f"overall_{metric}_{dataset_name}.csv")
            ).drop(columns="Unnamed: 0")
            df_res["IG"] = df_res["Unnamed: 1"].apply(
                lambda x: x.split(r"\_")[0] == "CONST"
            )
            df_res["Gen"] = df_res["Unnamed: 1"].apply(lambda x: x.split(r"\_")[-1])
            st.dataframe(
                df_res.set_index(["IG", "Gen"]).drop(columns="Unnamed: 1"),
                use_container_width=True
            )
            st.caption(f"Overall model performance, {metric}.")
        with col2:
            df_res = pd.read_csv(
                join(ANALYSIS_PATH, f"metric_disp_{dataset_name}_{attr}_{metric}.csv")
            ).drop(columns="Unnamed: 0")
            df_res["IG"] = df_res["Unnamed: 1"].apply(
                lambda x: x.split(r"\_")[0] == "CONST"
            )
            df_res["Gen"] = df_res["Unnamed: 1"].apply(lambda x: x.split(r"\_")[-1])
            st.dataframe(
                df_res.set_index(["IG", "Gen"]).drop(columns="Unnamed: 1"),
                use_container_width=True
            )
            st.caption(
                f"Disparities in model performance across sensitive groups, {metric} "
                f"metric, {attr} (advantaged / disadvantaged)."
            )

        # Plots
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                join(ANALYSIS_PATH, f"boxplots_{dataset_name}_{metric}_{attr}.png"),
            )
        with col2:
            st.image(
                join(ANALYSIS_PATH, f"perfdiff_{dataset_name}_{metric}.png"),
                caption=(
                    "Performance difference between Generation without and with "
                    "constraints (No IG - With IG). "
                    f"**{SENSITIVE_ATTRS[dataset_name][0]}**"
                )
            )
elif dataset_name == "SYNTHETIC":
    SYNTH_PATH = join(dirname(__file__), "qualitative_analysis", "figures")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            join(SYNTH_PATH, "overall_dist.png"),
        )
    with col2:
        st.image(
            join(SYNTH_PATH, "overall_dist_IC.png"),
        )
    st.write(
        "Score distribution of a model trained over synthetic data, tested on real data."
        " Left side: model trained without integrity constraints. Right side: model "
        "trained with integrity constraints."
    )
    st.image(join(SYNTH_PATH, "scores_distribution_cat.png"))
    st.write(
        "Feature importance distributions with and without integrity constraints."
    )
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            join(SYNTH_PATH, "waterfall_uncertain_pred.png"),
        )
    with col2:
        st.image(
            join(SYNTH_PATH, "waterfall_uncertain_pred_ic.png"),
        )
    st.write("Feature contributions over an observation with high uncertainty (left: no IC, right: with IC).")
    st.image(join(SYNTH_PATH, "contributions_sample.png"))
    st.write("Feature distributions per observation (sample of 200 observations).")
else:
    # Overall analysis
    summary = pd.read_csv(
        join(ANALYSIS_PATH, "datasets_summary.csv")
    ).set_index("Dataset name")
    st.dataframe(summary)
