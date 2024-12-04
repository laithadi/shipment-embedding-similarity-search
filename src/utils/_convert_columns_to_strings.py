import pandas as pd
import unittest

def convert_columns_to_string_with_column_name(df):
    """
    Converts all columns in a pandas DataFrame to string format, 
    concatenating the column name (cleaned) to each value.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with all values as strings,
                      prefixed by their respective column names.
    """
    def clean_column_name(col_name):
        # Replace underscores, dashes, or slashes with spaces and convert to lowercase
        return col_name.replace("_", " ").replace("-", " ").replace("/", " ").strip().lower()

    def convert_value(value, col_name):
        # Clean the column name
        clean_name = clean_column_name(col_name)
        if isinstance(value, pd.Timestamp):
            # Format dates as 'YYYY-MM-DD'
            value = value.strftime("%Y-%m-%d")
        return f"{clean_name} {value}"

    # Apply conversion to all values
    return df.apply(lambda col: col.map(lambda value: convert_value(value, col.name)))


class TestConvertColumnsToStringWithColumnName(unittest.TestCase):
    """
    Unit tests for the convert_columns_to_string_with_column_name function.
    """

    @classmethod
    def setUpClass(cls):
        """
        Sets up a DataFrame for testing.
        """
        data = {
            "Numeric_Column": [1, 2, 3],
            "Date_Column": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Text_Column": ["A", "B", "C"],
            "Mixed_Column": [1, "2", pd.Timestamp("2023-01-01")]
        }
        cls.df = pd.DataFrame(data)
        cls.df["Date_Column"] = pd.to_datetime(cls.df["Date_Column"])

    def test_column_name_added(self):
        """
        Tests if the column name is correctly concatenated to the values.
        """
        converted_df = convert_columns_to_string_with_column_name(self.df)
        self.assertEqual(converted_df["Numeric_Column"].iloc[0], "numeric column 1")
        self.assertEqual(converted_df["Date_Column"].iloc[0], "date column 2023-01-01")
        self.assertEqual(converted_df["Text_Column"].iloc[0], "text column A")
        self.assertEqual(converted_df["Mixed_Column"].iloc[2], "mixed column 2023-01-01")

    def test_all_columns_converted(self):
        """
        Tests if all columns are converted to strings.
        """
        converted_df = convert_columns_to_string_with_column_name(self.df)
        for col in converted_df.columns:
            self.assertTrue(
                all(isinstance(value, str) for value in converted_df[col]),
                f"Column {col} contains non-string values"
            )