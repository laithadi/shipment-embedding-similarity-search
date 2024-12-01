import pathlib
import pandas as pd 
import logging 
import numpy as np 

import sys
root_path = pathlib.Path.cwd()
sys.path.append(str(root_path))

from src.utils import process_user_input, filter_data_by_cols
from src.ai_utils import DistilBertTextEmbedding

# logger setup 
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# global params 
data_path = pathlib.Path.cwd() / "data" / "take_home_dataset.csv" 
user_input_path = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "user_queries.txt"
user_input_df_cols = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "df_cols.txt"

# model
model = DistilBertTextEmbedding()

if __name__=="__main__":

    # read data (csv) using pandas 
    logger.info(f"\tLoading data from {data_path}...")
    df_raw = pd.read_csv(
        filepath_or_buffer= data_path,
        delimiter= ";",
        )
    logger.info(f"\tDone loading the data. Data size: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.\n")
    
    # get the selected columns to filter the df 
    logger.info(f"\tExtracting the selected columns to filter by from the text file: {user_input_df_cols}...\n")
    selected_cols = process_user_input(
        user_input_path= user_input_df_cols
    )

    # filter the df
    logger.info(f"\tFiltering DataFrame to include only the selected columns: {selected_cols}")
    df = filter_data_by_cols(
        df= df_raw,
        cols= selected_cols,
    )
    logger.info(f"\tDone filtering the data. New data size: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # get the list of queries 
    logger.info(f"\tExtracting user queries from the text file: {user_input_path}...")
    user_input = process_user_input(
        user_input_path= user_input_path,
    )
    logger.info(f"\tDone extracting the user queries. There is a total of {len(user_input)} queries.\n")

    # process one query at a time from the user input 
    query_results = []
    for q in user_input:
        embedded_query = model.embed_text(q.lower())  # Embed the query
        best_match = None
        best_score = -1
        col_res = {}

        # Iterate over columns in the dataframe
        for column in df.columns:
            # Embed the column's unique values (e.g., 'Product_Category' values)
            column_values = df[column].unique()
            for value in column_values:
                embedded_value = model.embed_text(value.lower())  # Embed the column value

                # Compute cosine similarity (similarity score between query and value embedding)
                similarity_score = np.dot(embedded_query, embedded_value) / (np.linalg.norm(embedded_query) * np.linalg.norm(embedded_value))
                
                # Track the best match (highest similarity score)
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = (column, value, best_score)

            col_res[column] = best_match

        # iterate over best results from each column, and get the final result with the highest best_match score 
        overall_best_score = -1 
        overall_best_col = None  
        for c in list(col_res.keys()):
            column_name, value, best_score = best_match
            if best_score > overall_best_score:
                overall_best = best_score
                overall_best_col = c
        
        # get rows for the final result 
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
    
    print(query_results)