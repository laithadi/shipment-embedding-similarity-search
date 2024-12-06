import pandas as pd
import logging 
import unittest

__all__ = ["convert_date_columns"]

logger = logging.getLogger(__name__)

def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns in a DataFrame that seem to contain date-like values into datetime type.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with date-like columns converted to datetime type.
    """
    df = df.copy()  # avoid modifying the original DataFrame

    for col in df.columns:
        # check if a column is object or string-like and contains date-like values
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            try:
                # attempt to convert column to datetime
                converted_col = pd.to_datetime(df[col], errors="coerce")
                # only update column if it contains valid dates (at least one non-NaT value)
                if not converted_col.isna().all():
                    df[col] = converted_col
                    logger.info(f"Converted column '{col}' to datetime.")
            except Exception as e:
                logger.info(f"Could not convert column '{col}' to datetime: {e}")
    return df



class TestConvertDateColumns(unittest.TestCase):
    """
    Unit tests for the convert_date_columns function.
    """

    def setUp(self) -> None:
        """
        Sets up a DataFrame for testing.
        """
        self.data = {
            "Estimated_Arrival_Date": ["2023-07-05", "2023-07-04", "Invalid Date"],
            "Order_ID": [123, 456, 789],
            "Order_date": ["2023/01/15", "2023/01/16", "2023/01/17"],
            "Invalid_Date_Column": ["Some Text", "Other Text", "More Text"],
        }
        self.df = pd.DataFrame(self.data)

    def test_convert_valid_date_columns(self) -> None:
        """
        Tests if valid date-like columns are converted to datetime format.
        """
        df_cleaned = convert_date_columns(self.df)

        # Check if specific columns are converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_cleaned["Estimated_Arrival_Date"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_cleaned["Order_date"]))

        # Ensure non-date columns remain unchanged
        self.assertFalse(pd.api.types.is_datetime64_any_dtype(df_cleaned["Order_ID"]))

    def test_invalid_date_handling(self) -> None:
        """
        Tests if invalid date values are replaced with NaT.
        """
        df_cleaned = convert_date_columns(self.df)
        self.assertTrue(pd.isna(df_cleaned["Estimated_Arrival_Date"].iloc[2]))

    def test_no_date_columns(self) -> None:
        """
        Tests a DataFrame with no date-like columns.
        """
        data = {
            "Column1": ["A", "B", "C"],
            "Column2": [1, 2, 3],
        }
        df = pd.DataFrame(data)
        df_cleaned = convert_date_columns(df)

        # Assert that the DataFrame remains unchanged
        pd.testing.assert_frame_equal(df, df_cleaned)

    def test_no_unexpected_modifications(self) -> None:
        """
        Tests if non-date columns remain unchanged.
        """
        df_cleaned = convert_date_columns(self.df)

        # Assert that non-date columns are identical to the original
        pd.testing.assert_series_equal(df_cleaned["Order_ID"], self.df["Order_ID"])
        pd.testing.assert_series_equal(df_cleaned["Invalid_Date_Column"], self.df["Invalid_Date_Column"])
