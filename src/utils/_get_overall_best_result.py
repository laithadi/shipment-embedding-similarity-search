import unittest

from default_configs import (
    ORIGINAL_FILENAME_KEY,
    VALUE_KEY,
    SCORE_KEY,
)
from typing import Dict, Any, Optional

__all__ = ["get_overall_best_result"]

def get_overall_best_result(col_res: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Determines the overall best result across columns by finding the entry with the highest similarity score.

    Args:
        col_res (Dict[str, Dict[str, Any]]): A dictionary where:
            - Keys are column names.
            - Values are dictionaries containing:
                - ORIGINAL_FILENAME_KEY: The name of the original column.
                - VALUE_KEY: The best matching value.
                - SCORE_KEY: The highest similarity score.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing:
            - ORIGINAL_FILENAME_KEY: The column name with the best score.
            - VALUE_KEY: The corresponding value.
            - SCORE_KEY: The highest similarity score.
            Returns None if col_res is empty.
    """
    overall_best_score = -1
    overall_best_result = None

    for column, result in col_res.items():
        # Extract the score and compare
        best_score = result[SCORE_KEY]
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_result = result

    return overall_best_result


class TestGetOverallBestResult(unittest.TestCase):
    """
    Unit tests for the get_overall_best_result function.
    """

    def test_get_best_result(self):
        """
        tests if the function correctly identifies the best result.
        """
        col_res = {
            "Column1": ("Column1", "Value1", 0.8),
            "Column2": ("Column2", "Value2", 0.95),
            "Column3": ("Column3", "Value3", 0.85),
        }
        result = get_overall_best_result(col_res)
        self.assertEqual(result, ("Column2", "Value2", 0.95))

    def test_empty_col_res(self):
        """
        tests if the function returns None when col_res is empty.
        """
        col_res = {}
        result = get_overall_best_result(col_res)
        self.assertIsNone(result)

    def test_single_column(self):
        """
        tests if the function handles a single column correctly.
        """
        col_res = {"Column1": ("Column1", "Value1", 0.9)}
        result = get_overall_best_result(col_res)
        self.assertEqual(result, ("Column1", "Value1", 0.9))
