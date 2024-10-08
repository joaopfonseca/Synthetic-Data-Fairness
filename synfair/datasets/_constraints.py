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


IfLongDuration = create_custom_constraint_class(
    is_valid_fn=is_valid_duration,
)

IsAbove60 = create_custom_constraint_class(
    is_valid_fn=is_above_60,
    transform_fn=is_above_60_transform,
    reverse_transform_fn=is_above_60_transform,
)

custom_constraints_list = [
    "IfLongDuration",
    "IsAbove60",
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
            "constraint_class": "IsAbove60",
            "constraint_parameters": {"column_names": ["age", "age>60"]},
        },
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
}
