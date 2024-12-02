import requests
from urllib.parse import urljoin
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSTravelTime
from mlresearch.datasets.base import Datasets

from ._constraints import constraints

URL = "https://zenodo.org/records/14052137/files/"


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
            meta = eval(
                requests.get(
                    urljoin(URL, name.lower().replace(" ", "_") + "_dtypes.json")
                ).text
            )
            if "Label" in meta["tables"]["table"]["columns"]:
                meta["tables"]["table"]["columns"]["target"] = meta["tables"]["table"][
                    "columns"
                ]["Label"]
                meta["tables"]["table"]["columns"].pop("Label")

            self.metadata_[name] = meta

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
            name: len(
                [
                    col_name
                    for col_name, meta in (
                        data_types[name]["tables"]["table"]["columns"].items()
                    )
                    if meta["sdtype"] != "numerical"
                ]
            )
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
        data.target = data.target - 1
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
        data = data[
            (data["height"] > 100)
            & (data["height"] < 250)
            & (data["weight"] > 30)
            & (data["weight"] < 200)
            & (data["ap_hi"] > 60)
            & (data["ap_hi"] < 300)
            & (data["ap_lo"] > 30)
            & (data["ap_lo"] < 150)
            & (data["ap_hi"] > data["ap_lo"])
        ]
        return data

    def fetch_credit(self):
        """
        Fetch the Give Me Credit dataset.

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "credit-training.csv"))
        data.drop(columns=["Unnamed: 0"], inplace=True)
        data.rename(columns={"SeriousDlqin2yrs": "target"}, inplace=True)
        data.dropna(inplace=True)
        data["age>60"] = (data["age"] > 60).astype(int)
        data = data[
            (data["RevolvingUtilizationOfUnsecuredLines"] < 1.7)
            & (data["age"] >= 18)
            & (data["NumberOfTime30-59DaysPastDueNotWorse"] < 20)
            & (data["MonthlyIncome"] < 40000)
            & (data["NumberOfOpenCreditLinesAndLoans"] <= 40)
            & (data["NumberOfTimes90DaysLate"] < 20)
            & (data["NumberRealEstateLoansOrLines"] < 20)
            & (data["NumberOfTime60-89DaysPastDueNotWorse"] < 10)
            & (data["NumberOfDependents"] <= 10)
        ]
        data = data[
            data["NumberRealEstateLoansOrLines"]
            < data["NumberOfOpenCreditLinesAndLoans"]
        ]
        return data

    # def fetch_traveltime(self):
    #     """
    #     Fetch the TravelTime dataset (based on the Folktables project covering
    #     US census data).

    #     See https://zenodo.org/records/13375221 for more information.
    #     """
    #     data = pd.read_csv(urljoin(URL, "traveltime.csv"), index_col=1)
    #     data.drop(columns=["Unnamed: 0"], inplace=True)
    #     data.rename(columns={"LABEL": "target"}, inplace=True)
    #     data = data[data["POVPIP"] >= 0]
    #     return data

    def fetch_traveltime(self):
        """
        Fetch the TravelTime dataset (based on the Folktables project covering
        US census data).

        See https://arxiv.org/pdf/2108.04884 for more information.
        """
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )
        data = data_source.get_data(states=["NY"], download=True)
        data, labels, _ = ACSTravelTime.df_to_pandas(data)
        data.drop(columns=["ST", "PUMA", "POWPUMA"])
        data["target"] = labels
        data = data.astype(int)
        # data = data[data["POVPIP"] >= 0]
        return data

    def fetch_bank(self):
        """
        Fetch the Bank Marketing dataset.

        See https://zenodo.org/records/13375221 for more information.
        """
        dfs = [
            pd.read_csv(urljoin(URL, "bank-full.csv"), sep=";"),
            pd.read_csv(urljoin(URL, "bank.csv"), sep=";"),
        ]
        data = pd.concat(dfs)
        data.rename(columns={"y": "target"}, inplace=True)
        data = data[data["duration"] < 2000]

        # Someone with credit in default must have some type of loan/line of credit.
        all_credit_features = [data["balance"] < 0]
        for var in ["housing", "loan"]:
            mask = data[var] == "yes"
            all_credit_features.append(mask)

        default_mask = data["default"] == "yes"
        credit_mask = np.any(all_credit_features, axis=0)
        mask = ~(default_mask & ~credit_mask)
        data = data[mask]
        data["target"] = (data["target"] == "yes").astype(int)
        data = data[data["campaign"] <= 30]

        return data

    def fetch_law_school(self):
        """
        Fetch the Law School dataset.

        Preprocessing based on:
        - https://github.com/tailequy/fairness_dataset/tree/main

        Description:
        - https://arxiv.org/pdf/2110.00530.pdf
        - https://doi.org/10.1002/widm.1452

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "law_school.csv")).set_index("ID")
        data = data[
            [
                "decile1b",
                "decile3",
                "lsat",
                "ugpa",
                "zfygpa",
                "zgpa",
                "fulltime",
                "fam_inc",
                "male",
                "tier",
                "race",
                "pass_bar",
            ]
        ]
        data.dropna(inplace=True)
        for col_name in ["fulltime", "fam_inc", "male", "tier", "race"]:
            data[col_name] = data[col_name].astype(int)

        data.rename(columns={"pass_bar": "target"}, inplace=True)
        return data

    def fetch_diabetes(self):
        """
        Fetch the Diabetes dataset.

        Extracted from:
        - https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019

        See https://zenodo.org/records/13375221 for more information.
        """
        data = pd.read_csv(urljoin(URL, "diabetes.csv"))
        data.rename(columns={"Diabetic": "target"}, inplace=True)
        data["target"] = (data["target"] == "yes").astype(int)
        data.dropna(inplace=True)
        return data
