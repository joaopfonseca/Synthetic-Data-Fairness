from sdv.constraints import create_custom_constraint_class


def is_valid_duration(column_names, data):
    """Applicable for the German credit dataset."""
    # Duration (in month, converted to years) - Attribute2
    duration = column_names[0]
    # Age (in years) - Attribute13
    age = column_names[1]

    return data[duration] / 12 + data[age] < 80


# def transform(column_names, data):
#     duration = column_names[0]
#     age = column_names[1]
#
#     data[duration] = data[duration].mask(
#         data[duration] / 12 + data[age] < 80, 80 - data[age] * 12
#     )
#
#     return data

IfLongDuration = create_custom_constraint_class(
    is_valid_fn=is_valid_duration,
)

custom_constraints_list = [
    "IfLongDuration",
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
                # Installment rate in percentage of disposable income
                "column_name": "Attribute8",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Installment rate in percentage of disposable income
                "column_name": "Attribute8",
                "relation": "<=",
                "value": 4,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Present residence since
                "column_name": "Attribute11",
                "relation": ">=",
                "value": 1,
            },
        },
        {
            "constraint_class": "ScalarInequality",
            "constraint_parameters": {
                # Present residence since
                "column_name": "Attribute11",
                "relation": "<=",
                "value": 4,
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
}
