# Base
from os.path import join, dirname

# Models
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sdv.single_table import GaussianCopulaSynthesizer
from mlresearch.preprocessing import PipelineEncoder
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
            "OHE",
            PipelineEncoder(
                encoder=OneHotEncoder(
                    handle_unknown="infrequent_if_exist", sparse_output=False
                ),
            ),
            {},
        )
    ],
    "CLASSIFIERS": [
        (
            "LGBM",
            LGBMClassifier(verbose=-1),
            {
                "n_estimators": [100, 250, 500, 750],
                # "max_depth": np.arange(5, 25, step=5),
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
                "n_estimators": [100, 250, 500, 750],
                # "max_depth": np.arange(5, 25, step=5),
                # "min_samples_split": np.arange(2, 11, step=2),
                # "min_samples_leaf": np.arange(1, 11, step=2),
                # "max_features": ["sqrt", "log2"],
                # "bootstrap": [True, False],
            },
        ),
    ],
}

# NOTE: TURN THIS INTO A LOOP

# Set up data
dataset_name = DATASET_NAMES[0]
df = dict(objs["DATASETS"])[dataset_name]
metadata = dict(objs["METADATA"])[dataset_name]
constraints = dict(objs["CONSTRAINTS"])[dataset_name]

config_path = join(
    dirname(__file__), dataset_name.lower().replace(" ", "_") + "_config.yml"
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
    param = {"__".join(k.split("__")[1:]): v[0] for k, v in param.items()}
    param["OHE__features"] = categorical_columns
    clf = clone(dict(pipelines)[est_name])
    clf.set_params(**param)
    models_config[f"{est_name}__{param}"] = clf

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
