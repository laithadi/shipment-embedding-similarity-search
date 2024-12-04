import pandas as pd
from typing import Any
import unittest

def add_string_version_columns_with_column_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new columns to the DataFrame for non-string columns.
    Each new column contains string representations of the original values,
    prefixed with the cleaned column name and the column name prefixed by 's_'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with new columns added for non-string columns.
    """
    def clean_column_name(col_name: str) -> str:
        """
        Cleans a column name by replacing underscores, dashes, or slashes with spaces
        and converting it to lowercase.

        Args:
            col_name (str): The original column name.

        Returns:
            str: The cleaned column name.
        """
        return col_name.replace("_", " ").replace("-", " ").replace("/", " ").strip().lower()

    def convert_value(value: Any, col_name: str) -> str:
        """
        Converts a value to a string, prefixed by the cleaned column name.
        Formats dates in 'Month Day Year' format.

        Args:
            value (Any): The original value.
            col_name (str): The column name.

        Returns:
            str: The converted value with the column name prefix.
        """
        clean_name = clean_column_name(col_name)
        if isinstance(value, pd.Timestamp):
            # format dates as 'Month Day Year' (e.g., "January 21 2023")
            value = value.strftime("%B %d %Y")
        return f"{clean_name} {value}"

    # create a copy of the DataFrame to add new columns
    df = df.copy()

    for col in df.columns:
        # check if the column is not already a string column
        if not pd.api.types.is_string_dtype(df[col]):
            # generate the new column name
            new_col_name = f"s_{col}"
            # add the new column with string-converted values
            df[new_col_name] = df[col].map(lambda value: convert_value(value, col))

    return df


class TestAddStringVersionColumnsWithColumnName(unittest.TestCase):
    """
    Unit tests for the add_string_version_columns_with_column_name function.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up a DataFrame for testing.
        """
        data = {
            "Numeric_Column": [1, 2, 3],
            "Date_Column": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Text_Column": ["A", "B", "C"],
            "Mixed_Column": [1, "2", pd.Timestamp("2023-01-01")],
        }
        cls.df = pd.DataFrame(data)
        cls.df["Date_Column"] = pd.to_datetime(cls.df["Date_Column"])

    def test_new_columns_added(self) -> None:
        """
        Tests if new columns are added for non-string columns.
        """
        converted_df = add_string_version_columns_with_column_name(self.df)
        self.assertIn("s_Numeric_Column", converted_df.columns)
        self.assertIn("s_Date_Column", converted_df.columns)
        self.assertIn("s_Mixed_Column", converted_df.columns)
        self.assertNotIn("s_Text_Column", converted_df.columns)  # text column shouldn't have a new column

    def test_column_name_and_value_concatenation(self) -> None:
        """
        Tests if the column name is correctly concatenated to the values in the new columns.
        """
        converted_df = add_string_version_columns_with_column_name(self.df)
        # Check numeric column
        self.assertEqual(converted_df["s_Numeric_Column"].iloc[0], "numeric column 1")
        self.assertEqual(converted_df["s_Numeric_Column"].iloc[2], "numeric column 3")
        
        # Check date column (in word format)
        self.assertEqual(converted_df["s_Date_Column"].iloc[0], "date column January 01 2023")
        self.assertEqual(converted_df["s_Date_Column"].iloc[1], "date column January 02 2023")
        self.assertEqual(converted_df["s_Date_Column"].iloc[2], "date column January 03 2023")
        
        # Check mixed column
        self.assertEqual(converted_df["s_Mixed_Column"].iloc[0], "mixed column 1")
        self.assertEqual(converted_df["s_Mixed_Column"].iloc[2], "mixed column January 01 2023")

    def test_original_columns_unchanged(self) -> None:
        """
        Tests if the original columns remain unchanged.
        """
        converted_df = add_string_version_columns_with_column_name(self.df)
        pd.testing.assert_series_equal(converted_df["Numeric_Column"], self.df["Numeric_Column"])
        pd.testing.assert_series_equal(converted_df["Date_Column"], self.df["Date_Column"])
        pd.testing.assert_series_equal(converted_df["Text_Column"], self.df["Text_Column"])
        pd.testing.assert_series_equal(converted_df["Mixed_Column"], self.df["Mixed_Column"])

    def test_no_extra_columns_for_strings(self) -> None:
        """
        Tests that no new column is added for string columns.
        """
        converted_df = add_string_version_columns_with_column_name(self.df)
        self.assertNotIn("s_Text_Column", converted_df.columns)

    def test_date_format(self) -> None:
        """
        Tests if date columns are correctly formatted in the new columns.
        """
        converted_df = add_string_version_columns_with_column_name(self.df)
        self.assertEqual(converted_df["s_Date_Column"].iloc[0], "date column January 01 2023")
        self.assertEqual(converted_df["s_Date_Column"].iloc[2], "date column January 03 2023")