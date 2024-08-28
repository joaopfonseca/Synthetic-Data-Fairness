"""
Wrapper for SDV package to allow integration with sklearn pipelines.
"""

import os
import numpy as np
from imblearn.base import BaseSampler
from sdv.metadata import SingleTableMetadata

from ..datasets._constraints import custom_constraints_list


class SDVGenerator(BaseSampler):
    def __init__(self, model=None, n_rows=None, constraints=None, metadata=None):
        self.model = model
        self.n_rows = n_rows
        self.constraints = constraints
        self.metadata = metadata

    def fit(self, X, y=None):
        # Check y parameter (included for compatibility with sklearn)
        if y:
            X = np.hstack((X, y.reshape(-1, 1)))

        # Check n_rows value
        if self.n_rows is None:
            self.n_rows_ = X.shape[0]
        elif isinstance(self.n_rows, float):
            self.n_rows_ = int(self.n_rows * X.shape[0])
        else:
            self.n_rows_ = self.n_rows

        # Check constraints
        if self.constraints is None:
            self.constraints_ = []

        if self.metadata is None:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(X)
        else:
            metadata = self.metadata

        self.metadata_ = metadata
        self.model_ = self.model(self.metadata_)
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
        return self.model.sample(num_rows=self.n_rows_)

    def fit_resample(self, X, y=None):
        return self.fit(X, y).resample()
