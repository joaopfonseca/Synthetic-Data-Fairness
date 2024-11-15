# Base
from os.path import join, dirname, isfile
import yaml

# import numpy as np
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
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer

# from mlresearch.synthetic_data import GeometricSMOTE
from mlresearch.utils import check_pipelines
from mlresearch.model_selection import ModelSearchCV

# Own objects
from synfair.datasets import SynFairDatasets
from synfair.synthetic_data import SDVGenerator  # , ImbLearnGenerator
from synfair.metrics import make_fairness_metrics

DATASET_NAMES = [
    "BANK",
    "LAW SCHOOL",
    "DIABETES",
    "GERMAN CREDIT",
    "CARDIO",
    "CREDIT",
    # "TRAVELTIME",
]
RESULTS_PATH = join(dirname(__file__), "results")


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
            {"model__epochs": [200, 2000]},
        ),
        # (
        #     "LIN",
        #     ImbLearnGenerator(model=SMOTENC),
        #     {
        #         "model__k_neighbors": [3, 5],
        #     },
        # ),
        # (
        #     "GEOM",
        #     ImbLearnGenerator(model=GeometricSMOTE),
        #     {
        #         "model__k_neighbors": [3, 5],
        #         "model__selection_strategy": ["combined"],
        #         "model__truncation_factor": [-1.0, 0.0, 1.0],
        #         "model__deformation_factor": [0.0, 0.5, 1.0],
        #     },
        # ),
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
        # (
        #     "MLP",
        #     MLPClassifier(max_iter=10000),
        #     {
        #         "hidden_layer_sizes": [(100,), (50, 50), (25, 25, 25), (10, 10)],
        #         "alpha": [0.0001, 0.001, 0.01],
        #     },
        # ),
        (
            "DT",
            DecisionTreeClassifier(),
            {"criterion": ["gini", "entropy"], "max_depth": [5, 10]},
        ),
        # (
        #     "LGBM",
        #     LGBMClassifier(verbose=-1),
        #     {
        #         "n_estimators": [100, 250, 500, 750],
        #         "max_depth": np.arange(5, 20, step=4),
        #         # "num_leaves": np.arange(10, 50, step=10),
        #         # "learning_rate": np.logspace(-4, -1, 4),
        #         # "subsample": np.linspace(0.5, 1.0, 5),
        #         # "reg_lambda": np.logspace(-1, 2, 4),
        #     },
        # ),
        (
            "RF",
            RandomForestClassifier(),
            {
                "criterion": ["gini", "entropy"],
                "n_estimators": [10, 100, 1000],
                "max_depth": [5, 10],
                # "max_features": ["sqrt", "log2"],
                # "bootstrap": [True, False],
            },
        ),
    ],
    "SENSITIVE_ATTRS": yaml.safe_load(
        open(join(dirname(__file__), "sensitive_attrs.yml"))
    ),
    "SCORING": "f1_macro",
    "N_SPLITS": 5,
    "N_RUNS": 1,  # 3
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
