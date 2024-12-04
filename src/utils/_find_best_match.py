import numpy as np
import pandas as pd
import unittest

from src.playbooks.default_runner_configs import (
    ORIGINAL_FILENAME_KEY,
    VALUE_KEY,
    SCORE_KEY,
)
from typing import List, Tuple, Any


__all__ = ["find_best_match"]

def find_best_match(
    unique_col_values: List[Any],
    embedded_query: np.ndarray,
    column: str,
    df: pd.DataFrame,
    original_column: str,
    textual_model: Any
) -> dict:
    """
    Finds the best match from a list of unique column values by calculating cosine similarity with a query embedding.

    Args:
        unique_col_values (List[Any]): A list of unique values from a column.
        embedded_query (np.ndarray): The embedding vector of the query.
        column (str): The name of the current column being processed.
        df (pd.DataFrame): The DataFrame containing the data.
        original_column (str): The original column name (used if 's_' prefix is applied).
        textual_model (Any): The model used for embedding the textual values.

    Returns:
        dict: A dictionary containing:
            - ORIGINAL_FILENAME_KEY (str): The name of the original column.
            - VALUE_KEY (Any): The best matching value.
            - SCORE_KEY (float): The highest similarity score.
    """
    best_score = -1
    best_match = {
        ORIGINAL_FILENAME_KEY: None,
        VALUE_KEY: None,
        SCORE_KEY: None
    }

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
            # update the dictionary with new values
            best_match.update({
                ORIGINAL_FILENAME_KEY: original_column,
                VALUE_KEY: original_value,
                SCORE_KEY: best_score
            })

    return best_match


class TestFindBestMatch(unittest.TestCase):
    """
    Unit tests for the find_best_match function.
    """

    pass