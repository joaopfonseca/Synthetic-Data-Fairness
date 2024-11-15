"""
Wrapper for SDV package to allow integration with sklearn pipelines.
"""

import os

# from collections import Counter
import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from sdv.metadata import Metadata

from ..datasets._constraints import custom_constraints_list


class SDVGenerator(BaseSampler):
    def __init__(
        self,
        model=None,
        model_params=None,
        n_rows=None,
        constraints=None,
        metadata=None,
        random_state=None,
    ):
        self.model = model
        self.model_params = model_params
        self.n_rows = n_rows
        self.constraints = constraints
        self.metadata = metadata
        self.random_state = random_state

    def fit(self, X, y=None):
        # The only random seed reference I could find in SDV docs...
        # https://sdv.dev/blog/eng-sdv-constraints/
        np.random.seed(self.random_state)

        # print("::NAME::", self.model, self.model_params)
        # print("::ORIGINAL::", Counter(y))
        # if len(Counter(y)) < 2:
        #     print("::DEBUG:: Only one class in the dataset.")

        # Check y parameter (included for compatibility with sklearn)
        if y is not None:
            X = pd.concat([X, y], axis=1)
            self._y_name = X.columns[-1]

        # Check n_rows value
        if self.n_rows is None:
            self.n_rows_ = X.shape[0]
        elif isinstance(self.n_rows, float):
            self.n_rows_ = int(self.n_rows * X.shape[0])
        else:
            self.n_rows_ = self.n_rows

        # Check constraints
        self.constraints_ = [] if self.constraints is None else self.constraints

        # Check metadata
        metadata = Metadata()
        if self.metadata is None:
            metadata.detect_from_dataframe(X)
        else:
            metadata = metadata.load_from_dict(self.metadata)

        self.metadata_ = metadata
        self.model_ = self.model(self.metadata_, **self.model_params)
        filepath = os.path.dirname(os.path.abspath(__file__))
        self.model_.load_custom_constraint_classes(
            filepath=os.path.join(
                filepath, os.path.pardir, "datasets", "_constraints.py"
            ),
            class_names=custom_constraints_list,
        )
        self.model_.add_constraints(self.constraints_)
        self.model_.fit(X)
        return self

    def resample(self, X=None, y=None):
        X_res = self.model_.sample(num_rows=self.n_rows_)

        # print("::RESAMPLED::", Counter(X_res[self._y_name]))
        # if len(Counter(X_res[self._y_name])) < 2:
        #     print("::DEBUG::", self.model_.__class__.__name__)

        if hasattr(self, "_y_name"):
            y_res = X_res[self._y_name]
            X_res = X_res.drop(columns=self._y_name)

            return X_res, y_res
        return X_res

    def fit_resample(self, X, y=None):
        return self.fit(X, y).resample()

    def set_params(self, **params):
        base_params = {}
        model_params = {}
        for key, value in params.items():
            if key in self.get_params():
                base_params[key] = value

            if key.startswith("model__"):
                model_params[key.split("__")[-1]] = value

        base_params["model_params"] = model_params

        return super().set_params(**base_params)

    def _fit_resample(self, X, y):
        pass
