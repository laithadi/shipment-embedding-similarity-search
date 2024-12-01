import logging
import numpy as np
import pandas as pd
import pathlib
import sys

# append the root path to system paths for relative imports
root_path = pathlib.Path.cwd()
sys.path.append(str(root_path))

from src.ai_utils import DistilBertTextEmbedding
from src.utils import filter_data_by_cols, process_user_input, data_cols_types

# logger setup
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# global parameters
data_path = pathlib.Path.cwd() / "data" / "take_home_dataset.csv"
user_input_path = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "user_queries.txt"
user_input_df_cols = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "df_cols.txt"

# model initialization
model = DistilBertTextEmbedding()

if __name__ == "__main__":
    """
    Main script for processing user queries and filtering data based on embeddings.

    This script:
    1. Loads a CSV dataset.
    2. Filters the dataset by selected columns from a text file.
    3. Reads user queries from a text file.
    4. Uses a DistilBERT-based model to process the queries.
    5. Computes cosine similarity between query embeddings and dataset column values.
    6. Outputs the best-matched results for each query, including row IDs.
    """

    # step 1: load data
    logger.info(f"\tLoading data from {data_path}...")
    df_raw = pd.read_csv(
        filepath_or_buffer=data_path,
        delimiter=";",
    )
    # TODO: remove the first column
    logger.info(f"\tDone loading the data. Data size: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.\n")
    
    # step 2: get the selected columns to filter the DataFrame
    logger.info(f"\tExtracting the selected columns to filter by from the text file: {user_input_df_cols}...\n")
    selected_cols = process_user_input(user_input_path=user_input_df_cols)

    # filter the DataFrame
    logger.info(f"\tFiltering DataFrame to include only the selected columns: {selected_cols}")
    df = filter_data_by_cols(df=df_raw, cols=selected_cols)
    logger.info(f"\tDone filtering the data. New data size: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # step 3: get the list of queries
    logger.info(f"\tExtracting user queries from the text file: {user_input_path}...")
    user_input = process_user_input(user_input_path=user_input_path)
    logger.info(f"\tDone extracting the user queries. There is a total of {len(user_input)} queries.\n")

    # step 4: process one query at a time
    query_results = []
    for q in user_input:
        embedded_query = model.embed_text(q.lower())  # embed the query
        best_match = None
        best_score = -1
        col_res = {}

        # iterate over columns in the DataFrame
        for column in df.columns:

            # TODO: using data_cols_types, embed column values based on type

            # embed the column's unique values (e.g., 'Product_Category' values)
            column_values = df[column].unique()
            for value in column_values:
                embedded_value = model.embed_text(value.lower())  # embed the column value

                # compute cosine similarity between query and column value embeddings
                similarity_score = np.dot(embedded_query, embedded_value) / (
                    np.linalg.norm(embedded_query) * np.linalg.norm(embedded_value)
                )
                
                # track the best match (highest similarity score)
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = (column, value, best_score)

            col_res[column] = best_match

        # determine the overall best result across columns
        overall_best_score = -1
        overall_best_col = None
        for c in list(col_res.keys()):
            column_name, value, best_score = best_match
            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_col = c
        
        # retrieve rows for the final result
        f_col_name, f_value, f_best_score = best_match
        matching_rows = df[df[f_col_name] == f_value].index.tolist()
        adjusted_rows = [index + 2 for index in matching_rows]

        # store the results for this query into the final query_results list
        query_results.append({
            "column_name": column_name,
            "value": value,
            "row_ids": [f'row{row}' for row in adjusted_rows],
            "best_score": f_best_score
        })
    
    # output query results
    print(query_results)
