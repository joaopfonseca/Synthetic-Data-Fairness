from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.base import BaseSampler


class ImbLearnGenerator(BaseSampler):
    def __init__(
        self,
        model=None,
        model_params=None,
        n_rows=None,
        metadata=None,
        random_state=None,
    ):
        self.model = model
        self.model_params = model_params
        self.n_rows = n_rows
        self.metadata = metadata
        self.random_state = random_state

    def fit(self, X, y):

        # Check n_rows value
        if self.n_rows is None:
            self.n_rows_ = X.shape[0]
        elif isinstance(self.n_rows, float):
            self.n_rows_ = int(self.n_rows * X.shape[0])
        else:
            self.n_rows_ = self.n_rows

        # Check categorical features
        self.categorical_features_ = [
            sdtype_dict["sdtype"] == "categorical"
            for col, sdtype_dict in self.metadata["columns"].items()
            if col != "target"
        ]

        # Check sampling strategy
        self._counts = Counter(y)
        ratios = {label: count / y.shape[0] for label, count in self._counts.items()}
        self.sampling_strategy_ = {
            key: self._counts[key] + int(value * self.n_rows_)
            for key, value in ratios.items()
        }

        # Check model params
        self.model_params = self.model_params or {}

        self.model_ = self.model(
            categorical_features=self.categorical_features_,
            sampling_strategy=self.sampling_strategy_,
            random_state=self.random_state,
            **self.model_params
        )
        self.model_.fit(X, y)

        return self

    def resample(self, X, y):
        n_rows = X.shape[0]
        X_new, y_new = self.fit_resample(self._X, self._y)
        X_new, _, y_new, _ = train_test_split(
            X_new,
            y_new,
            train_size=n_rows,
            random_state=self.random_state,
            stratify=y_new,
        )
        return X_new, y_new

    def fit_resample(self, X, y):
        self.fit(X, y)

        # Save data for resampling later on
        self._X = X
        self._y = y

        X_res, y_res = self.model_.fit_resample(X, y)
        X_new, y_new = [], []
        for class_label, n_original_samples in self._counts.items():
            label_mask = y_res == class_label
            X_new.append(X_res[label_mask][n_original_samples:])
            y_new.append(y_res[label_mask][n_original_samples:])
        X_new = np.concatenate(X_new)
        y_new = np.concatenate(y_new)
        return pd.DataFrame(X_new, columns=X.columns), pd.Series(y_new, name=y.name)

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
        """A placeholder. It is overriden by the self.fit_resample method."""
        pass
