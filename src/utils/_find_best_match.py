import numpy as np
import pandas as pd
from typing import List, Tuple, Any
import unittest


__all__ = ["find_best_match"]

def find_best_match(
    unique_col_values: List[Any],
    embedded_query: np.ndarray,
    column: str,
    df: pd.DataFrame,
    original_column: str,
    textual_model
) -> Tuple[str, Any, float]:
    """
    Finds the best match from a list of unique column values by calculating cosine similarity with a query embedding.

    Args:
        unique_col_values (List[Any]): A list of unique values from a column.
        embedded_query (np.ndarray): The embedding vector of the query.
        column (str): The name of the current column being processed.
        df (pd.DataFrame): The DataFrame containing the data.
        original_column (str): The original column name (used if 's_' prefix is applied).
        textual_model: The model used for embedding the textual values.

    Returns:
        Tuple[str, Any, float]: A tuple containing:
            - The original column name.
            - The best matching value.
            - The highest similarity score.
    """
    best_score = -1
    best_match = (original_column, None, best_score)

    for value in unique_col_values:
        # embed the column value
        embedded_value = textual_model.embed_text(value.lower())

        # compute cosine similarity between query and column value embeddings
        similarity_score = np.dot(embedded_query, embedded_value) / (
            np.linalg.norm(embedded_query) * np.linalg.norm(embedded_value)
        )

        # map back to the original value if using an "s_" column
        if column.startswith("s_"):
            original_value = df[original_column][df[column] == value].iloc[0]
        else:
            original_value = value

        # track the best match
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = (original_column, original_value, best_score)

    return best_match


class TestFindBestMatch(unittest.TestCase):
    """
    Unit tests for the find_best_match function.
    """

    pass