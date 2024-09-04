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
    ],
    "CARDIO": [
        {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "ap_lo",
                "high_column_name": "ap_hi",
                "strict_boundaries": True,  # greater than
            },
        }
    ],
    "CREDIT": [
        {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "NumberRealEstateLoansOrLines",
                "high_column_name": "NumberOfOpenCreditLinesAndLoans",
                "strict_boundaries": False,  # greater or equal than
            },
        }
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
