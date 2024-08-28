import requests
import json
from urllib.parse import urljoin
import pandas as pd
from mlresearch.datasets.base import Datasets

from ._constraints import constraints

URL = "https://zenodo.org/records/13385610/files/"


class SynFairDatasets(Datasets):
    """Datasets used to do research on fairness in synthetic data generation."""
    def download(self):
        """Download the datasets."""
        # Download the data
        super().download(keep_index=True)
        content = []
        for name, dataset in self.content_:
            content.append((name, dataset.set_index(dataset.columns[0])))
        self.content_ = content

        # Download the metadata
        self.metadata_ = {}
        for name, _ in self.content_:
            self.metadata_[name] = json.loads(
                requests.get(
                    urljoin(URL, name.lower().replace(" ", "_")+"_dtypes.json")
                ).text
            )

        # Download the constraints
        self.constraints_ = {}
        for name, _ in self.content_:
            try:
                self.constraints_[name] = constraints[name]
            except KeyError:
                raise ValueError(f"Constraints for dataset {name} not found.")

        return self

    def summarize_datasets(self):
        summary = super().summarize_datasets()
        constraints = {name: len(const) for name, const in self.constraints_.items()}
        summary["Constraints"] = constraints

        data_types = self.metadata_
        non_metric = {
            name: len([
                col_name
                for col_name, meta in data_types[name]["columns"].items()
                if meta["sdtype"] != "numerical"
            ])
            for name in data_types
        }
        summary["Non-metric"] = non_metric

        return summary

    def metadata(self):
        """Return metadata for all datasets."""
        return self.metadata_

    def constraints(self):
        """Return denial constraints for all datasets."""
        return self.constraints_

    def fetch_german_credit(self):
        """
        Fetch the German credit dataset.

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "german_credit.csv"))
        data.drop(columns=["Unnamed: 0"], inplace=True)
        data.rename(columns={"Label": "target"}, inplace=True)
        data.index.name = "ID"
        return data

    def fetch_cardio(self):
        """
        Fetch the Cardio dataset.

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "cardio_train.csv"), sep=";", index_col=0)
        data["age"] = data["age"] / 365.25
        data.rename(columns={"cardio": "target"}, inplace=True)

        # Drop tuples with invalid values
        data = data[data["ap_hi"] > data["ap_lo"]]
        return data

    def fetch_credit(self):
        """
        Fetch the Give Me Credit dataset.

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "cs-training.csv"))
        data.drop(columns=["Unnamed: 0"], inplace=True)
        data.rename(columns={"SeriousDlqin2yrs": "target"}, inplace=True)
        data.dropna(inplace=True)
        return data

    def fetch_traveltime(self):
        """
        Fetch the TravelTime dataset (based on the Folktables project covering
        US census data).

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "folktables_processed_data.csv"), index_col=1)
        data.drop(columns=["Unnamed: 0"], inplace=True)
        data.rename(columns={"LABEL": "target"}, inplace=True)
        data = data[data["POVPIP"] >= 0]
        return data
