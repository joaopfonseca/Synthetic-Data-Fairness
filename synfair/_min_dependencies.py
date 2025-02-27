"""All minimum dependencies for synthetic-data-fairness."""

import argparse

ML_RESEARCH_MIN_VERSION = "0.5.1"
# NUMPY_MIN_VERSION = "1.20.0"
PANDAS_MIN_VERSION = "2.1.0"
SDV_MIN_VERSION = "1.17.1"
PYYAML_MIN_VERSION = "6.0.2"
# SKLEARN_MIN_VERSION = "1.2.0"
# IMBLEARN_MIN_VERSION = "0.8.0"
# TQDM_MIN_VERSION = "4.46.0"
# MATPLOTLIB_MIN_VERSION = "2.2.3"

# The values are (version_spec, comma separated tags)
dependent_packages = {
    "ml-research": (ML_RESEARCH_MIN_VERSION, "install"),
    "pandas": (PANDAS_MIN_VERSION, "install"),
    "sdv": (SDV_MIN_VERSION, "install"),
    "pyyaml": (PYYAML_MIN_VERSION, "install"),
    # "numpy": (NUMPY_MIN_VERSION, "install"),
    # "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
    # "imbalanced-learn": (IMBLEARN_MIN_VERSION, "install"),
    # "requests": ("2.26.0", "install"),
    # "tqdm": (TQDM_MIN_VERSION, "optional"),
    # "matplotlib": (MATPLOTLIB_MIN_VERSION, "optional"),
    # "pytest-cov": ("3.0.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "black": ("22.3", "tests"),
    # "pylint": ("2.12.2", "tests"),
    # "mypy": ("1.6.1", "tests"),
    # "types-requests": ("2.31.0.10", "tests"),
    # "coverage": ("6.2", "tests"),
    # "sphinx": ("4.2.0", "docs"),
    # "numpydoc": ("1.0.0", "docs, tests"),
    # "sphinx-material": ("0.0.35", "docs"),
    # "nbsphinx": ("0.8.7", "docs"),
    # "recommonmark": ("0.7.1", "docs"),
    # "sphinx-markdown-tables": ("0.0.15", "docs"),
    # "sphinx-copybutton": ("0.4.0", "docs"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests", "all"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))
    tag_to_packages["all"].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
