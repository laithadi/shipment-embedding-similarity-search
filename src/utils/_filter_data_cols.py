import pandas as pd

__all__ = ["filter_data_by_cols"]

def filter_data_by_cols(df, cols):
    """
    Filters a DataFrame to include only the specified columns.

    This function checks if all requested columns exist in the DataFrame. If the `cols` list is empty,
    it returns the original DataFrame. If any columns are missing, it raises a KeyError.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list): A list of column names to retain in the filtered DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing only the specified columns.

    Raises:
        KeyError: If any of the specified columns are not in the DataFrame.
    """
    # edge case: return the original DataFrame if the columns list is empty
    if len(cols) == 0:
        return df

    # check if all requested columns are in the DataFrame
    missing_columns = [col for col in cols if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following columns are not in the DataFrame: {missing_columns}")

    # return the filtered DataFrame with only the specified columns
    return df[cols]
