import numpy as np
from sdv.constraints import create_custom_constraint_class


def is_valid_duration(column_names, data):
    """Applicable for the German credit dataset."""
    # Duration (in month, converted to years) - Attribute2
    duration = column_names[0]
    # Age (in years) - Attribute13
    age = column_names[1]

    return data[duration] / 12 + data[age] < 80


def is_above_60(column_names, data):
    """Applicable for the Credit dataset."""
    # Age in years
    age = column_names[0]
    age_dummy = column_names[1]

    return (data[age] > 60) == data[age_dummy]


def is_above_60_transform(column_names, data):
    """Applicable for the Credit dataset."""
    # Age in years
    age = column_names[0]
    age_dummy = column_names[1]

    data[age_dummy] = (data[age] > 60).astype(int)
    return data


def credit_default_has_credit(column_names, data):
    """
    Applicable for the Bank dataset.

    Reasoning: Someone with credit in default must have some type of loan/line of credit.
    """
    default_credit = column_names[0]  # Default
    balance = column_names[1]  # Balance
    credit_vars = column_names[2:]  # Housing + Loan

    all_credit_features = [data[balance] < 0]
    for var in credit_vars:
        mask = data[var] == "yes"
        all_credit_features.append(mask)

    default_mask = data[default_credit] == "yes"
    credit_mask = np.any(all_credit_features, axis=0)
    mask = ~(default_mask & ~credit_mask)
    return mask


def previous_contacts(column_names, data):
    previous_outcome = column_names[0]
    previous_contacts = column_names[1]
    return ~(
        data[previous_outcome].isin(["failure", "success"]) & data[previous_contacts]
        > 1
    )


IfLongDuration = create_custom_constraint_class(
    is_valid_fn=is_valid_duration,
)

IsAbove60 = create_custom_constraint_class(
    is_valid_fn=is_above_60,
    transform_fn=is_above_60_transform,
    reverse_transform_fn=is_above_60_transform,
)

IfDefaultHasCredit = create_custom_constraint_class(
    is_valid_fn=credit_default_has_credit,
)

ContactedWithOutcome = create_custom_constraint_class(
    is_valid_fn=previous_contacts,
)

custom_constraints_list = [
    "IfLongDuration",
    "IsAbove60",
    "IfDefaultHasCredit",
    "ContactedWithOutcome",
]

constraints = {
    "GERMAN CREDIT": [
        {
            "constraint_class": "IfLongDuration",
            "constraint_parameters": {"column_names": ["Attribute2", "Attribute13"]},
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "Attribute2",  # Duration in month
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "Attribute2",  # Duration in month
                "relation": "<=",
                "value": 75,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Credit amount
                "column_name": "Attribute5",
                "relation": ">=",
                "value": 250,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Credit amount
                "column_name": "Attribute5",
                "relation": "<=",
                "value": 18500,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Age in years
                "column_name": "Attribute13",
                "relation": ">=",
                "value": 19,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Age in years
                "column_name": "Attribute13",
                "relation": "<=",
                "value": 75,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Number of existing credits at this bank
                "column_name": "Attribute16",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Number of existing credits at this bank
                "column_name": "Attribute16",
                "relation": "<=",
                "value": 4,
            },
        },
    ],
    "CARDIO": [
        {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "ap_lo",
                "high_column_name": "ap_hi",
                "strict_boundaries": True,  # greater than
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "age",
                "relation": ">=",
                "value": 29,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "age",
                "relation": "<=",
                "value": 65,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "height",
                "relation": ">=",
                "value": 100,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "height",
                "relation": "<=",
                "value": 250,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "weight",
                "relation": ">=",
                "value": 30,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "weight",
                "relation": "<=",
                "value": 200,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "ap_hi",
                "relation": ">=",
                "value": 60,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "ap_hi",
                "relation": "<=",
                "value": 300,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "ap_lo",
                "relation": ">=",
                "value": 30,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "ap_lo",
                "relation": "<=",
                "value": 150,
            },
        },
    ],
    "CREDIT": [
        {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "NumberRealEstateLoansOrLines",
                "high_column_name": "NumberOfOpenCreditLinesAndLoans",
                "strict_boundaries": False,  # greater or equal than
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "RevolvingUtilizationOfUnsecuredLines",
                "relation": "<",
                "value": 1.7,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "age",
                "relation": ">=",
                "value": 18,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "NumberOfTime30-59DaysPastDueNotWorse",
                "relation": "<",
                "value": 20,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "MonthlyIncome",
                "relation": "<",
                "value": 40000,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "NumberOfOpenCreditLinesAndLoans",
                "relation": "<=",
                "value": 40,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "NumberOfTimes90DaysLate",
                "relation": "<",
                "value": 20,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "NumberRealEstateLoansOrLines",
                "relation": "<",
                "value": 20,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "NumberOfTime60-89DaysPastDueNotWorse",
                "relation": "<",
                "value": 10,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "NumberOfDependents",
                "relation": "<=",
                "value": 10,
            },
        },
    ],
    "TRAVELTIME": [
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "AGEP",
                "relation": ">=",
                "value": 16,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "AGEP",
                "relation": "<=",
                "value": 95,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "POVPIP",
                "relation": ">=",
                "value": 0,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "POVPIP",
                "relation": "<=",
                "value": 501,
            },
        },
    ],
    "BANK": [
        {
            "constraint_class": "IfDefaultHasCredit",
            "constraint_parameters": {
                "column_names": ["default", "balance", "housing", "loan"]
            },
        },
        {
            "constraint_class": "ContactedWithOutcome",
            "constraint_parameters": {"column_names": ["poutcome", "previous"]},
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "age",
                "relation": ">=",
                "value": 18,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "age",
                "relation": "<=",
                "value": 95,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "day",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "day",
                "relation": "<=",
                "value": 31,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "campaign",
                "relation": "<=",
                "value": 30,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "campaign",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "pdays",
                "relation": ">=",
                "value": -1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "pdays",
                "relation": "<=",
                "value": 871,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "previous",
                "relation": ">=",
                "value": 0,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "previous",
                "relation": "<=",
                "value": 275,
            },
        },
    ],
    "LAW SCHOOL": [
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "decile1b",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "decile1b",
                "relation": "<=",
                "value": 10,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "decile3",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "decile3",
                "relation": "<=",
                "value": 10,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "lsat",
                "relation": ">=",
                "value": 11,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "decile3",
                "relation": "<=",
                "value": 48,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "ugpa",
                "relation": ">=",
                "value": 1.5,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "ugpa",
                "relation": "<=",
                "value": 4,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "zfygpa",
                "relation": ">=",
                "value": -3.35,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "zfygpa",
                "relation": "<=",
                "value": 3.25,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "zgpa",
                "relation": ">=",
                "value": -6.44,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "zgpa",
                "relation": "<=",
                "value": 3.45,
            },
        },
    ],
    "DIABETES": [
        {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "SoundSleep",
                "high_column_name": "Sleep",
                "strict_boundaries": False,  # greater or equal than
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "BMI",
                "relation": ">=",
                "value": 15,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "BMI",
                "relation": "<=",
                "value": 45,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "Sleep",
                "relation": ">=",
                "value": 4,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "Sleep",
                "relation": "<=",
                "value": 11,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "SoundSleep",
                "relation": ">=",
                "value": 0,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                "column_name": "SoundSleep",
                "relation": "<=",
                "value": 11,
            },
        },
    ],
}
