import pandas as pd

__all__ = ["filter_data_by_cols"]

def filter_data_by_cols(df, cols):
    # edge case: the cols list is empty 
    if len(cols) == 0:
        return df 


    # Check if all requested columns are in the DataFrame
    missing_columns = [col for col in cols if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following columns are not in the DataFrame: {missing_columns}")
    
    return df[cols]