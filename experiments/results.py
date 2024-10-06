# Base
from os.path import join, dirname, isfile
import numpy as np
import pandas as pd

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
from sklearn.model_selection import StratifiedKFold
# from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTENC
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from mlresearch.synthetic_data import GeometricSMOTE
from mlresearch.utils import check_pipelines
# from mlresearch.metrics import get_scorer
from mlresearch.model_selection import HalvingModelSearchCV

# Experimental design
from virny.datasets.base import BaseDataLoader
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.utils.custom_initializers import create_config_obj
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config

# Own objects
from synfair.datasets import SynFairDatasets
from synfair.synthetic_data import SDVGenerator, ImbLearnGenerator

DATASET_NAMES = [
    # "GERMAN CREDIT",
    # "CARDIO",
    # "CREDIT",
    "TRAVELTIME",
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
                    "truncnorm",
                    "uniform",
                    "gamma",
                    # "gaussian_kde",  # Raising memory issues
                ]
            },
        ),
        (
            "TVAE",
            SDVGenerator(model=TVAESynthesizer),
            {"model__epochs": [300, 600, 900]},
        ),
        (
            "LIN",
            ImbLearnGenerator(model=SMOTENC),
            {

                "model__k_neighbors": [3, 5],
            },
        ),
        (
            "GEOM",
            ImbLearnGenerator(model=GeometricSMOTE),
            {

                "model__k_neighbors": [3, 5],
                "model__selection_strategy": ["combined", "minority", "majority"],
                "model__truncation_factor": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "model__deformation_factor": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
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
            {"penalty": [None, "l1", "l2"], "solver": ["saga"]},
        ),
        ("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 6, 9, 12, 15]}),
        (
            "MLP",
            MLPClassifier(max_iter=10000),
            {
                "hidden_layer_sizes": [(100,), (50, 50), (25, 25, 25), (10, 10)],
                "alpha": [0.0001, 0.001, 0.01],
            },
        ),
        (
            "DT",
            DecisionTreeClassifier(),
            {"criterion": ["gini", "entropy"], "max_depth": np.arange(5, 20, step=5)},
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
                "n_estimators": [100, 250, 500],
                "max_depth": np.arange(5, 20, step=5),
                # "min_samples_split": np.arange(2, 11, step=3),
                # "min_samples_leaf": np.arange(2, 11, step=3),
                "max_features": ["sqrt", "log2"],
                # "bootstrap": [True, False],
            },
        ),
    ],
    "SCORING": "f1_macro",
    "N_SPLITS": 5,
    "N_RUNS": 1,  # 3
    "VIRNY_TEST_SET_FRACTION": 0.2,
    "RANDOM_STATE": 42,
    "N_JOBS": 4  # -1,
}

# Run experiments for each dataset
for dataset_name in DATASET_NAMES:

    # Set up data
    df = dict(CONFIG["DATASETS"])[dataset_name]
    metadata = dict(CONFIG["METADATA"])[dataset_name]
    constraints = dict(CONFIG["CONSTRAINTS"])[dataset_name]

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
        test_set_fraction=CONFIG["VIRNY_TEST_SET_FRACTION"],
        dataset_split_seed=CONFIG["RANDOM_STATE"],
    )

    # Set up models
    pipelines, params = check_pipelines(
        CONFIG["GENERATORS"],
        CONFIG["ENCODER"],
        CONFIG["CLASSIFIERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_runs=CONFIG["N_RUNS"],
    )

    # Add fixed params for preprocessing current dataset
    params_ = []
    for param in params:
        est_name = param["est_name"][0]
        clf = clone(dict(pipelines)[est_name])
        gen_name = est_name.split("|")[0]
        param[f"{est_name}__PREP__transformers"] = [[
            (
                "OHE",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", sparse_output=False
                ),
                categorical_columns,
            ),
        ]]

        metadata_params = [
            f"{est_name}__{p}" for p in clf.get_params()
            if p.split("__")[-1] == "metadata"
            and p.split("__")[0] == gen_name
        ]
        for mp in metadata_params:
            param[mp] = [metadata]

        params_.append(param)

    params = params_

    # Add version with constraints (if applicable)
    const_params = []
    for param in params:
        est_name = param["est_name"][0]
        gen_name = est_name.split("|")[0]
        clf = clone(dict(pipelines)[est_name])

        const_name = f"CONST_{est_name}"
        constraint_params = [
            f"{const_name}__{p}" for p in clf.get_params()
            if p.split("__")[-1] == "constraints"
            and p.split("__")[0] == gen_name
        ]
        const_param = {
            k.replace(est_name, const_name): v
            for k, v in param.items()
        }
        for cp in constraint_params:
            const_param["est_name"] = [const_name]
            const_param[cp] = [constraints]
            const_params.append(const_param)
            if const_name not in dict(pipelines):
                pipelines.append(
                    (const_name, clf)
                )
    params += const_params

    # Check if results already exist
    filename = join(RESULTS_PATH, f"param_tuning_{dataset_name}.pkl")
    if not isfile(filename):
        # Run parameter tuning
        experiment = HalvingModelSearchCV(
            estimators=pipelines,
            param_grids=params,
            factor=2,
            scoring=CONFIG["SCORING"],
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
        pd.DataFrame(experiment.cv_results_).to_pickle(
            filename
        )

    # Load results
    # results = pd.read_pickle(filename)
    # results = (
    #     results[
    #         ["param_est_name", "params", "mean_test_f1_macro"]
    #     ]
    #     .groupby("param_est_name")
    #     .apply(
    #         lambda df: df.loc[df["mean_test_f1_macro"].idxmax()],
    #         include_groups=False
    #     )
    # )
    # opt_param_dict = results["params"].to_dict()

    # models_config = {}
    # for param in opt_param_dict.values():
    #     est_name = param.pop("est_name")
    #     clf = clone(dict(pipelines)[est_name])
    #     param = {"__".join(k.split("__")[1:]): v for k, v in param.items()}

    #     clf.set_params(**param)
    #     models_config[est_name] = clf

    # # Set up fairness analysis
    # metrics_dct = compute_metrics_with_config(
    #     base_flow_dataset,
    #     config,
    #     models_config,
    #     RESULTS_PATH,
    # )
