# Base
from os.path import join, dirname
import numpy as np

# Models
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sdv.single_table import GaussianCopulaSynthesizer
from mlresearch.utils import check_pipelines

# Experimental design
from virny.datasets.base import BaseDataLoader
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.utils.custom_initializers import create_config_obj
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config

# Own objects
from synfair.datasets import SynFairDatasets
from synfair.synthetic_data import SDVGenerator

RNG_SEED = 42
TEST_SET_FRACTION = 0.2
N_RUNS = 1
DATASET_NAMES = ["GERMAN CREDIT"]
RESULTS_PATH = join(dirname(__file__), "results")


datasets = SynFairDatasets(names=DATASET_NAMES).download()

objs = {
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
                    "truncnorm",
                    "uniform",
                    "gamma",
                    "gaussian_kde",
                ]
            },
        ),
    ],
    "ENCODER": [
        (
            "PREP",
            ColumnTransformer(
                transformers=None,
                remainder=StandardScaler(),
            ),
            {},
        )
    ],
    "CLASSIFIERS": [
        ("CONSTANT", DummyClassifier(strategy="prior"), {}),
        (
            "LR",
            LogisticRegression(max_iter=10000),
            {"penalty": ["none", "l1", "l2"], "solver": ["saga"]},
        ),
        ("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 6, 9, 12]}),
        (
            "MLP",
            MLPClassifier(),
            {
                "hidden_layer_sizes": [(100,), (50, 50), (25, 25, 25)],
                "alpha": [0.0001, 0.001, 0.01],
            },
        ),
        (
            "DT",
            DecisionTreeClassifier(),
            {"criterion": ["gini", "entropy"], "max_depth": np.arange(5, 25, step=5)},
        ),
        (
            "LGBM",
            LGBMClassifier(verbose=-1),
            {
                "n_estimators": [100, 250, 500, 750],
                "max_depth": np.arange(5, 25, step=5),
                # "num_leaves": np.arange(10, 50, step=10),
                # "learning_rate": np.logspace(-4, -1, 4),
                # "subsample": np.linspace(0.5, 1.0, 5),
                # "reg_lambda": np.logspace(-1, 2, 4),
            },
        ),
        (
            "RF",
            RandomForestClassifier(),
            {
                "criterion": ["gini", "entropy"],
                "n_estimators": [100, 250, 500, 750],
                "max_depth": np.arange(5, 25, step=5),
                # "min_samples_split": np.arange(2, 11, step=2),
                # "min_samples_leaf": np.arange(1, 11, step=2),
                # "max_features": ["sqrt", "log2"],
                # "bootstrap": [True, False],
            },
        ),
    ],
}

# Set up data
for dataset_name in DATASET_NAMES:
    df = dict(objs["DATASETS"])[dataset_name]
    metadata = dict(objs["METADATA"])[dataset_name]
    constraints = dict(objs["CONSTRAINTS"])[dataset_name]

    config_path = join(
        dirname(__file__),
        "virny_data_configs",
        dataset_name.lower().replace(" ", "_") + "_config.yml"
    )
    config = create_config_obj(config_yaml_path=config_path)

    numerical_columns = [
        col
        for col, meta in metadata["columns"].items()
        if (meta["sdtype"] == "numerical" and col != "target")
    ]

    categorical_columns = [
        col
        for col, meta in metadata["columns"].items()
        if (meta["sdtype"] == "categorical" and col != "target")
    ]

    data_loader = BaseDataLoader(
        full_df=df,
        target="target",
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    base_flow_dataset = preprocess_dataset(
        data_loader=data_loader,
        column_transformer=FunctionTransformer(),
        sensitive_attributes_dct=config.sensitive_attributes_dct,
        test_set_fraction=TEST_SET_FRACTION,
        dataset_split_seed=RNG_SEED,
    )

    # Set up models
    pipelines, params = check_pipelines(
        objs["GENERATORS"],
        objs["ENCODER"],
        objs["CLASSIFIERS"],
        random_state=RNG_SEED,
        n_runs=N_RUNS,
    )

    models_config = {}
    for param in params:
        est_name = param.pop("est_name")[0]
        clf = clone(dict(pipelines)[est_name])
        param = {"__".join(k.split("__")[1:]): v[0] for k, v in param.items()}

        param_str = (
            str(param)
            .replace("'", "")
            .replace("{", "")
            .replace("}", "")
            .replace(" ", "")
        )

        param["PREP__transformers"] = [
            (
                "OHE",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", sparse_output=False
                ),
                categorical_columns,
            ),
        ]

        clf.set_params(**param)
        est_name_final = est_name.replace("|PREP", "")
        models_config[f"{est_name_final}__{param_str}"] = clf

        # Add version with constraints (if applicable)
        clf = clone(dict(pipelines)[est_name])
        param_names = list(clf.get_params().keys())
        if not est_name.startswith("NONE"):
            constraint_param_loc = [
                p.split("__")[-1] == "constraints" for p in param_names
            ]
            if True in constraint_param_loc:
                constraint_param = param_names[constraint_param_loc.index(True)]
                param[constraint_param] = constraints

                clf.set_params(**param)
                est_name_final = "CONST_" + est_name.replace("|PREP", "")
                models_config[f"{est_name_final}__{param_str}"] = clf

    # Set up experiment
    metrics_dct = compute_metrics_with_config(
        base_flow_dataset,
        config,
        models_config,
        RESULTS_PATH,
    )

    # NOTE:
    # - n_rows: same as original dataset
    # - Add more generators
