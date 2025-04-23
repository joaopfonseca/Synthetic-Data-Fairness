import pandas as pd
from sdv.constraints import create_custom_constraint_class


def comb_greater_than(column_names, data, value=None, or_equal=True, operation="+"):
    col1, col2 = column_names
    comb = eval(f"data[col1] {operation} data[col2]")
    
    if value is None:
        value = comb.min()
        
    validity = eval(
        "comb" + (">=" if or_equal else ">") + "value" 
    )
    return pd.Series(validity)


def comb_less_than(column_names, data, value=None, or_equal=True, operation="+"):
    col1, col2 = column_names
    comb = eval(f"data[col1] {operation} data[col2]")
    
    if value is None:
        value = comb.max()
        
    validity = eval(
        "comb" + ("<=" if or_equal else "<") + "value" 
    )
    return pd.Series(validity)


CombGreaterThan = create_custom_constraint_class(is_valid_fn=comb_greater_than)
CombLessThan = create_custom_constraint_class(is_valid_fn=comb_less_than)
