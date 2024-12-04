import logging
import numpy as np
import pandas as pd
import pathlib
import sys

# append the root path to system paths for relative imports
root_path = pathlib.Path.cwd()
sys.path.append(str(root_path))

from src.ai_utils import DistilBertTextEmbedding
from src.utils import (
    filter_data_by_cols, 
    process_user_input, 
    data_cols_types, 
    get_column_type,
    add_string_version_columns_with_column_name,
    )

# logger setup
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# global parameters
data_path = pathlib.Path.cwd() / "data" / "take_home_dataset.csv"
user_input_path = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "user_queries.txt"
user_input_df_cols = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "df_cols.txt"

# model initialization
textual_model = DistilBertTextEmbedding()
numerical_model = ...
date_model = ...

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
    df_filtered_cols = filter_data_by_cols(df=df_raw, cols=selected_cols)
    df = add_string_version_columns_with_column_name(df= df_filtered_cols)
    logger.info(f"\tDone filtering the data. New data size: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # step 3: get the list of queries
    logger.info(f"\tExtracting user queries from the text file: {user_input_path}...")
    user_input = process_user_input(user_input_path=user_input_path)
    logger.info(f"\tDone extracting the user queries. There is a total of {len(user_input)} queries.\n")

    # step 4: process one query at a time
    query_results = []
    for q in user_input:
        embedded_query = textual_model.embed_text(q.lower())  # embed the query
        best_match = None
        best_score = -1
        col_res = {}
        processed_columns = set()  # keep track of processed columns to avoid duplicates

        # iterate over all columns in the DataFrame
        for column in df.columns:

            if column.startswith("s_"):  # for "s_" columns, use the original column name
                original_column = column[2:]  # Strip "s_" prefix to get the original column name
                if original_column in processed_columns:  # skip if already processed
                    continue  
            else:
                original_column = column
                if f"s_{original_column}" in df.columns:  # skip original column if "s_" exists
                    continue 

            # embed the columns unique values
            column_values = df[column].unique()
            for value in column_values:
                embedded_value = textual_model.embed_text(value.lower())  # embed the column value

                # compute cosine similarity between query and column value embeddings
                similarity_score = np.dot(embedded_query, embedded_value) / (
                    np.linalg.norm(embedded_query) * np.linalg.norm(embedded_value)
                )
                
                # Map back to the original value if using an "s_" column
                if column.startswith("s_"):
                    # Find the original value in the non-stringified column
                    original_value = df[original_column][df[column] == value].iloc[0]
                else:
                    original_value = value

                # Track the best match (use the original column name and value)
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = (original_column, original_value, best_score) 

            # mark the column as processed
            processed_columns.add(original_column)
            col_res[original_column] = best_match  # Use the original column name as the key

        # determine the overall best result across columns
        overall_best_score = -1
        overall_best_col = None
        for c in list(col_res.keys()):
            column_name, value, best_score = col_res[c]
            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_col = c

        # retrieve rows for the final result using the original column name
        f_col_name, f_value, f_best_score = col_res[overall_best_col]
        matching_rows = df[df[f_col_name] == f_value].index.tolist()
        adjusted_rows = [index + 2 for index in matching_rows]

        # store the results for this query into the final query_results list
        query_results.append({
            "column_name": f_col_name,
            "value": int(f_value) if isinstance(f_value, (np.integer, int)) else float(f_value) if isinstance(f_value, (np.floating, float)) else f_value,  # Convert numpy numbers to Python types,
            "row_ids": [f'row{row}' for row in adjusted_rows],
            "best_score": float(f_best_score)
        })
    
    # output query results
    print(query_results)
