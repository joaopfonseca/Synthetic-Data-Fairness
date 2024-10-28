import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics._scorer import _Scorer


class FilteredScorer(_Scorer):
    """filter_feature_value is a tuple of (feature_name, value) to filter
    the data before prediction."""

    def __init__(
        self,
        score_func,
        sign=1,
        filter_feature_value=None,
        response_method="predict",
        **kwargs,
    ):
        self._score_func = score_func
        self._filter_feature_value = filter_feature_value
        self._kwargs = kwargs
        self._sign = sign
        self._response_method = response_method

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if self._filter_feature_value is not None:
            feature_name, value = self._filter_feature_value
            if isinstance(value, str):
                mask = X[:, feature_name] == value
            else:
                mask = np.isin(X[:, feature_name], value)
            X = X[mask]
            y_true = y_true[mask]

        y_pred = estimator.predict(X)

        return self._score_func(y_true, y_pred)

    def set_score_request(self):
        """
        Placeholder to overwrite sklearn's ``_BaseScorer.set_score_request`` function.
        It is not used and was raising a docstring error with scikit-learn v1.3.0.

        Note
        ----
        This placeholder will be removed soon
        """
        pass


def _get_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    return tp, tn, fp, fn


def false_positive_rate(y_true, y_pred, target_label=1):
    tp, tn, fp, fn = _get_conf_matrix(y_true, y_pred)
    fpr = fp / (fp + tn)
    return fpr


def false_negative_rate(y_true, y_pred, target_label=1):
    tp, tn, fp, fn = _get_conf_matrix(y_true, y_pred)
    fnr = fn / (fn + tp)
    return fnr


def overall_accuracy(y_true, y_pred, target_label=1):
    tp, tn, fp, fn = _get_conf_matrix(y_true, y_pred)
    acc = (tp + tn) / (tp + fp + fn + tn)
    return acc


def selection_rate(y_true, y_pred, target_label=1):
    tp, tn, fp, fn = _get_conf_matrix(y_true, y_pred)
    sr = (tp + fp) / (tp + fp + fn + tn)
    return sr


def positive_predictive_value(y_true, y_pred, target_label=1):
    """
    Positive Predictive Value (PPV) or Precision
    """
    tp, tn, fp, fn = _get_conf_matrix(y_true, y_pred)
    ppv = tp / (tp + fp)
    return ppv


def group_size(y_true, y_pred):
    return len(y_true)


def make_fairness_metrics(X, sensitive_attributes):
    """
    Create fairness metrics for a given dataset.

    sensitive_attributes must be a dictionary with the following structure:
    {
        "sensitive_attribute_1": ["minority_value_1", "minority_value_2"],
        "sensitive_attribute_2": ["minority_value_1", "minority_value_2"],
        ...
    }

    majority_group and minority_group are the values of the sensitive attribute, which
    can be passed as a list of values.
    """
    metrics = {
        "fnr": false_negative_rate,
        "fpr": false_positive_rate,
        "f1": f1_score,
        "accuracy": overall_accuracy,
        "selection_rate": selection_rate,
        "ppv": positive_predictive_value,
        "size": group_size,
    }
    fairness_metrics = {}

    # Fetches each sensitive attribute
    for sensitive_attribute, dis_values in sensitive_attributes.items():
        adv_values = np.unique(X[~np.isin(X[sensitive_attribute], dis_values)])

        # Fetches each metric
        for metric_name, score_func in metrics.items():

            # Set up metric for the disadvantaged + advantaged group
            for group, values in zip(["dis", "adv"], [dis_values, adv_values]):
                group_metric_name = f"{metric_name}_{sensitive_attribute}_{group}"
                metric = FilteredScorer(
                    score_func,
                    filter_feature_value=(sensitive_attribute, values),
                )
                fairness_metrics[group_metric_name] = metric

    metrics = {
        metric_name: FilteredScorer(score_func)
        for metric_name, score_func in metrics.items()
    }

    return {**metrics, **fairness_metrics}
