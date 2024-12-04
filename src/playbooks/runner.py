import json
import logging
import numpy as np
import pandas as pd
import pathlib
import sys

# append the root path to system paths for relative imports
root_path = pathlib.Path.cwd()
sys.path.append(str(root_path))

from src.ai_utils import DistilBertTextEmbedding
from src.playbooks.default_runner_configs import (
    DATA_PATH,
    USER_QUERY_INPUT,
    USER_DF_COLS_INPUT,
    OUTPUT_FILE_PATH,
    SIMILARITY_CALC_RES_PATH,
    ORIGINAL_FILENAME_KEY,
    VALUE_KEY,
    SCORE_KEY,
)
from src.utils import (
    add_string_version_columns_with_column_name,
    filter_data_by_cols,
    find_best_match,
    get_overall_best_result,
    process_user_input,
    NumpyEncoder,
)

# logger setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# model initialization
textual_model = DistilBertTextEmbedding()
numerical_model = ...  # to be implemented - IGNORE 
date_model = ...       # to be implemented - IGNORE 



if __name__ == "__main__":
    """
    main script for processing user queries and filtering data based on embeddings.

    this script:
    1. loads a CSV dataset.
    2. filters the dataset by selected columns from a text file.
    3. reads user queries from a text file.
    4. uses a DistilBERT-based model to process the queries.
    5. computes cosine similarity between query embeddings and dataset column values.
    6. outputs the best-matched results for each query, including row IDs.
    """

    # step 1: load data
    logger.info(f"loading data from {DATA_PATH}...")
    df_raw = pd.read_csv(
        filepath_or_buffer= DATA_PATH,
        delimiter= ";",
    )
    logger.info(f"data loaded successfully. Data size: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    # step 2: get the selected columns to filter the DataFrame
    logger.info(f"extracting the selected columns to filter by from {USER_DF_COLS_INPUT}...")
    selected_cols = process_user_input(user_input_path= USER_DF_COLS_INPUT)

    # filter the DataFrame
    logger.info(f"filtering DataFrame to include only the selected columns: {selected_cols}")
    df_filtered_cols = filter_data_by_cols(df= df_raw, cols= selected_cols)
    df = add_string_version_columns_with_column_name(df= df_filtered_cols)
    logger.info(f"data filtered successfully. New data size: {df.shape[0]} rows, {df.shape[1]} columns")

    # step 3: get the list of queries
    logger.info(f"extracting user queries from {USER_QUERY_INPUT}...")
    user_input = process_user_input(user_input_path= USER_QUERY_INPUT)
    logger.info(f"user queries extracted successfully. Total queries: {len(user_input)}")

    # step 4: process one query at a time
    query_results = []
    detailed_record = {}
    for q in user_input:
        logger.info(f"now processing user query: {q}")  
        embedded_query = textual_model.embed_text(q.lower())  # embed the query
        col_res = {}
        processed_columns = set()  # keep track of processed columns to avoid duplicates

        # iterate over all columns in the DataFrame
        for column in df.columns:
            logger.info(f"calculating similarities for {column} column...")  
            if column.startswith("s_"):  # for "s_" columns, use the original column name
                original_column = column[2:]  # strip "s_" prefix to get the original column name
                if original_column in processed_columns:  # skip if already processed
                    continue
            else:
                original_column = column
                if f"s_{original_column}" in df.columns:  # skip original column if "s_" exists
                    continue

            # columns unique values
            column_values = df[column].unique()

            # step 5: find the queries best match from the columns unique values 
            best_match = find_best_match(
                unique_col_values= column_values,
                embedded_query= embedded_query,
                column= column,
                df= df,
                original_column= original_column,
                textual_model= textual_model,
            )

            # mark the column as processed
            processed_columns.add(original_column)
            col_res[original_column] = best_match  # use the original column name as the key
            detailed_record[q] = col_res

        # determine the overall best result across columns
        overall_best_result = get_overall_best_result(col_res)
        if overall_best_result:
            f_col_name, f_value, f_best_score = (
                overall_best_result[ORIGINAL_FILENAME_KEY],
                overall_best_result[VALUE_KEY],
                overall_best_result[SCORE_KEY],
            )
            matching_rows = df[df[f_col_name] == f_value].index.tolist()
            adjusted_rows = [index + 2 for index in matching_rows]

            # store the results for this query into the final query_results list
            query_results.append({
                "column_name": f_col_name,
                "value": int(f_value) if isinstance(f_value, (np.integer, int)) else float(f_value) if isinstance(f_value, (np.floating, float)) else f_value,  # convert numpy numbers to Python types
                "row_ids": [f'row{row}' for row in adjusted_rows],
                "best_score": float(f_best_score),
                "user_query": q,
            })

    # step 6: output the final results
    logger.info(f"query results: {query_results}")

    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(query_results, f, indent=4)

    logger.info(f"query results saved to {OUTPUT_FILE_PATH}")

    # serialize the data - ignore this, i had to hack at it 'till my json displayed nicely in the file
    ser_detailed_record = str(json.dumps(detailed_record, cls= NumpyEncoder, indent= 4)).strip()
    with open(SIMILARITY_CALC_RES_PATH, "w") as f:
        f.write(ser_detailed_record)
    logger.info(f"similarity calculations saved to {SIMILARITY_CALC_RES_PATH}")
