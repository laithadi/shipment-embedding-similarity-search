import pandas as pd
import unittest
from typing import Union


__all__ = ["get_column_type"]

def get_column_type(df: pd.DataFrame, column_name: str) -> str:
    """
    Determines the type of a column in a pandas DataFrame using match-case.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to check.

    Returns:
        str: The type of the column ('textual', 'numeric', 'date', or 'unknown').
    """
    col = df[column_name]

    match True:
        case _ if pd.api.types.is_numeric_dtype(col):
            return "numeric"
        case _ if pd.api.types.is_datetime64_any_dtype(col):
            return "date"
        case _ if pd.api.types.is_string_dtype(col):
            return "textual"
        case _:
            return "unknown"


class TestGetColumnType(unittest.TestCase):
    """
    unit tests for the get_column_type function.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        sets up a DataFrame for testing.
        """
        # create a test DataFrame with various column types
        data = {
            "Priority": ["High", "Low", "Medium"],
            "Days_from_shipment_to_delivery": [2, 5, 3],
            "Start_Shipping_Date": ["2023-01-01", "2023-01-05", "2023-01-07"],
            "Unknown_Column": [None, None, None],
        }
        cls.df = pd.DataFrame(data)
        cls.df["Start_Shipping_Date"] = pd.to_datetime(cls.df["Start_Shipping_Date"])

    def test_textual_column(self) -> None:
        """
        tests if textual columns are correctly identified.
        """
        self.assertEqual(get_column_type(self.df, "Priority"), "textual")

    def test_numeric_column(self) -> None:
        """
        tests if numeric columns are correctly identified.
        """
        self.assertEqual(get_column_type(self.df, "Days_from_shipment_to_delivery"), "numeric")

    def test_date_column(self) -> None:
        """
        tests if date columns are correctly identified.
        """
        self.assertEqual(get_column_type(self.df, "Start_Shipping_Date"), "date")

    def test_unknown_column(self) -> None:
        """
        tests if columns with unknown types are correctly identified.
        """
        self.assertEqual(get_column_type(self.df, "Unknown_Column"), "unknown")
