# Base
import argparse
from os.path import join, dirname, isfile
import yaml
import pickle
from copy import deepcopy
from itertools import product

# Core
import numpy as np
import pandas as pd

# Models
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # , FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# from lightgbm import LGBMClassifier
# from imblearn.over_sampling import SMOTENC
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
    CTGANSynthesizer,
)

# from mlresearch.synthetic_data import GeometricSMOTE
from mlresearch.utils import check_pipelines
from mlresearch.model_selection import ModelSearchCV

# Own objects
from synfair.datasets import SynFairDatasets
from synfair.synthetic_data import SDVGenerator  # , ImbLearnGenerator
from synfair.metrics import make_fairness_metrics

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset name", default=None)

dataset_name = parser.parse_args().dataset
if dataset_name is None:
    DATASET_NAMES = [
        "DIABETES",
        "GERMAN CREDIT",
        "LAW SCHOOL",
        "BANK",
        "CARDIO",
        "CREDIT",
        # "TRAVELTIME",
    ]
else:
    DATASET_NAMES = [dataset_name]

RESULTS_PATH = join(dirname(__file__), "results")
ANALYSIS_PATH = join(dirname(__file__), "analysis")


datasets = SynFairDatasets(names=DATASET_NAMES).download()

CONFIG = {
    "DATASETS": list(datasets),
    "METADATA": list(datasets.metadata().items()),
    "CONSTRAINTS": list(datasets.constraints().items()),
    "GENERATORS": [
        ("NONE", None, {}),
        (
            "GC",
            SDVGenerator(model=GaussianCopulaSynthesizer),
            {
                "model__default_distribution": [
                    "norm",
                    "beta",
                    "uniform",
                ]
            },
        ),
        (
            "TVAE",
            SDVGenerator(model=TVAESynthesizer),
            {"model__epochs": [2000, 20000]},
        ),
        (
            "CTGAN",
            SDVGenerator(model=CTGANSynthesizer),
            {"model__epochs": [2000, 20000]},
        ),
    ],
    "ENCODER": [
        (
            "PREP",
            ColumnTransformer(
                transformers=None,
                remainder=StandardScaler(),
                force_int_remainder_cols=False,
            ),
            {},
        )
    ],
    "CLASSIFIERS": [
        ("CONSTANT", DummyClassifier(strategy="prior"), {}),
        (
            "LR",
            LogisticRegression(max_iter=10000),
            {"penalty": ["l1", "l2"], "solver": ["saga"], "C": [0.1, 1.0]},
        ),
        ("KNN", KNeighborsClassifier(), {"n_neighbors": [1, 5, 10]}),
        (
            "DT",
            DecisionTreeClassifier(),
            {"criterion": ["gini", "entropy"], "max_depth": [5, 10]},
        ),
        (
            "RF",
            RandomForestClassifier(),
            {
                "criterion": ["gini", "entropy"],
                "n_estimators": [10, 100, 1000],
                "max_depth": [5, 10],
            },
        ),
    ],
    "SENSITIVE_ATTRS": yaml.safe_load(
        open(join(dirname(__file__), "sensitive_attrs.yml"))
    ),
    "N_SPLITS": 5,
    "N_RUNS": 1,
    "RANDOM_STATE": 42,
    "N_JOBS": -1,
}


def add_dataset_param(pipelines, params, categorical_columns, metadata):
    """
    Add dataset-specific parameters to the pipelines and parameter grid.
    """
    params_ = []
    for param in params:
        est_name = param["est_name"][0]
        clf = clone(dict(pipelines)[est_name])
        gen_name = est_name.split("|")[0]
        param[f"{est_name}__PREP__transformers"] = [
            [
                (
                    "OHE",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist", sparse_output=False
                    ),
                    categorical_columns,
                ),
            ]
        ]

        metadata_params = [
            f"{est_name}__{p}"
            for p in clf.get_params()
            if p.split("__")[-1] == "metadata" and p.split("__")[0] == gen_name
        ]
        for mp in metadata_params:
            param[mp] = [metadata]

        params_.append(param)

    return pipelines, params_


def add_constraints(pipelines, params, constraints):
    """
    Add constraints to the pipelines and parameter grid.
    """
    const_pipelines = []
    const_params = []
    for param in params:
        est_name = param["est_name"][0]
        gen_name = est_name.split("|")[0]
        clf = clone(dict(pipelines)[est_name])

        const_name = f"CONST_{est_name}"
        constraint_params = [
            f"{const_name}__{p}"
            for p in clf.get_params()
            if p.split("__")[-1] == "constraints" and p.split("__")[0] == gen_name
        ]
        const_param = {k.replace(est_name, const_name): v for k, v in param.items()}
        for cp in constraint_params:
            const_param["est_name"] = [const_name]
            const_param[cp] = [constraints]
            const_params.append(const_param)
            if const_name not in dict(const_pipelines):
                const_pipelines.append((const_name, clf))
    return const_pipelines, const_params


def get_best_params(df_results, target_metric="mean_test_f1"):
    params_ = df_results.groupby("param_est_name").apply(
        lambda x: x.iloc[x[target_metric].argmax()], include_groups=False
    )["params"]
    return list(params_)


def set_pipeline_parameters(pipelines, parameters):
    pipelines_ = []
    for params in parameters:
        params = deepcopy(params)
        est_name = params.pop("est_name")
        params = {
            key.replace(f"{est_name}__", ""): value for key, value in params.items()
        }
        pipeline = clone(dict(pipelines)[est_name]).set_params(**params)
        pipelines_.append((est_name, pipeline))

    return pipelines_


def synthetic_size_parameters(pipelines, rows_perc, param_name="n_rows"):
    rows_params = []
    for name, pipeline in pipelines:
        n_rows_params = [
            p_ for p_ in pipeline.get_params().keys() if p_.endswith(param_name)
        ]
        for n_rows, perc in product(n_rows_params, rows_perc):
            param_ = {"est_name": [name], f"{name}__{n_rows}": [perc]}
            rows_params.append(param_)
    return rows_params


# Run experiments for each dataset
for dataset_name in DATASET_NAMES:

    # Set up data
    df = dict(CONFIG["DATASETS"])[
        dataset_name
    ]  # .sample(n=200, random_state=42)  # uncomment for testing purposes
    metadata = dict(CONFIG["METADATA"])[dataset_name]
    constraints = dict(CONFIG["CONSTRAINTS"])[dataset_name]
    sensitive_attributes = CONFIG["SENSITIVE_ATTRS"][dataset_name]

    numerical_columns = [
        col
        for col, meta in metadata["tables"]["table"]["columns"].items()
        if (meta["sdtype"] == "numerical" and col != "target")
    ]

    categorical_columns = [
        col
        for col, meta in metadata["tables"]["table"]["columns"].items()
        if (meta["sdtype"] == "categorical" and col != "target")
    ]

    # Set up models
    pipelines, params = check_pipelines(
        CONFIG["GENERATORS"],
        CONFIG["ENCODER"],
        CONFIG["CLASSIFIERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_runs=CONFIG["N_RUNS"],
    )

    # Add fixed params for preprocessing current dataset
    pipelines, params = add_dataset_param(
        pipelines, params, categorical_columns, metadata
    )

    # Add version with constraints (if applicable)
    const_pipelines, const_params = add_constraints(pipelines, params, constraints)
    pipelines += const_pipelines
    params += const_params

    # setup fairness metrics
    scoring = make_fairness_metrics(df, sensitive_attributes)
    scoring_synth = make_fairness_metrics(df, sensitive_attributes, use_synthetic=True)
    scoring = {**scoring, **scoring_synth}

    #####################################################################################
    # RQ1
    # Check if results already exist
    filename = join(RESULTS_PATH, f"param_tuning_{dataset_name}.pkl")
    if not isfile(filename):
        # Run parameter tuning
        experiment = ModelSearchCV(
            estimators=pipelines,
            param_grids=params,
            scoring=scoring,
            n_jobs=CONFIG["N_JOBS"],
            cv=StratifiedKFold(
                n_splits=CONFIG["N_SPLITS"],
                shuffle=True,
                random_state=CONFIG["RANDOM_STATE"],
            ),
            verbose=1,
            return_train_score=True,
            refit=False,
        ).fit(df.drop(columns=["target"]), df["target"])

        # Save results
        pd.DataFrame(experiment.cv_results_).to_pickle(filename)

    # Read results and set up best models
    df_results = pickle.load(open(filename, "rb"))
    best_params = get_best_params(df_results, target_metric="mean_test_f1")
    pipelines_ = set_pipeline_parameters(pipelines, best_params)

    # Set up parameters for varying number of rows
    rows_perc = np.arange(0.1, 2.1, 0.1)
    rows_params = synthetic_size_parameters(pipelines_, rows_perc)

    #####################################################################################
    # RQ2
    # Check if results already exist
    filename = join(RESULTS_PATH, f"synth_size_{dataset_name}.pkl")
    if not isfile(filename):
        # Run parameter tuning
        experiment = ModelSearchCV(
            estimators=pipelines_,
            param_grids=rows_params,
            scoring=scoring,
            n_jobs=CONFIG["N_JOBS"],
            cv=StratifiedKFold(
                n_splits=CONFIG["N_SPLITS"],
                shuffle=True,
                random_state=CONFIG["RANDOM_STATE"],
            ),
            verbose=1,
            return_train_score=True,
            refit=False,
        ).fit(df.drop(columns=["target"]), df["target"])

        # Save results
        pd.DataFrame(experiment.cv_results_).to_pickle(filename)
