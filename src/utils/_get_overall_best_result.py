import unittest

from typing import Dict, Tuple, Any

__all__ = ["get_overall_best_result"]

def get_overall_best_result(col_res: Dict[str, Tuple[str, Any, float]]) -> Tuple[str, Any, float]:
    """
    Determines the overall best result across columns by finding the entry with the highest similarity score.

    Args:
        col_res (Dict[str, Tuple[str, Any, float]]): A dictionary where:
            - Keys are column names.
            - Values are tuples containing (column_name, value, similarity_score).

    Returns:
        Tuple[str, Any, float]: A tuple containing:
            - The column name with the best score.
            - The corresponding value.
            - The highest similarity score.
    """
    overall_best_score = -1
    overall_best_result = None

    for column, (column_name, value, best_score) in col_res.items():
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_result = (column_name, value, overall_best_score)

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
