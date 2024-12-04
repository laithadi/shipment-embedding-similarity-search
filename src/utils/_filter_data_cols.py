import pandas as pd
import unittest

from typing import List


__all__ = ["filter_data_by_cols"]

def filter_data_by_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Filters a DataFrame to include only the specified columns.

    This function checks if all requested columns exist in the DataFrame. If the `cols` list is empty,
    it returns the original DataFrame. If any columns are missing, it raises a KeyError.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (List[str]): A list of column names to retain in the filtered DataFrame.

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


class TestFilterDataByCols(unittest.TestCase):
    """
    Unit tests for the filter_data_by_cols function.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        sets up a sample DataFrame for testing.
        """
        cls.df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        })

    def test_filter_columns_valid(self) -> None:
        """
        tests filtering with valid columns.
        """
        filtered_df = filter_data_by_cols(self.df, ["A", "B"])
        self.assertEqual(list(filtered_df.columns), ["A", "B"])

    def test_empty_column_list(self) -> None:
        """
        tests returning the original DataFrame when the column list is empty.
        """
        filtered_df = filter_data_by_cols(self.df, [])
        pd.testing.assert_frame_equal(filtered_df, self.df)

    def test_missing_columns(self) -> None:
        """
        tests raising a KeyError when some columns are missing.
        """
        with self.assertRaises(KeyError):
            filter_data_by_cols(self.df, ["A", "D"])

    def test_all_columns(self) -> None:
        """
        tests filtering with all available columns.
        """
        filtered_df = filter_data_by_cols(self.df, ["A", "B", "C"])
        pd.testing.assert_frame_equal(filtered_df, self.df)
