import pathlib
import pandas as pd 
import logging 

import sys
root_path = pathlib.Path.cwd()
sys.path.append(str(root_path))

from src.utils import process_user_input, filter_data_by_cols

# logger setup 
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# global params 
data_path = pathlib.Path.cwd() / "data" / "take_home_dataset.csv" 
user_input_path = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "user_queries.txt"
user_input_df_cols = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "df_cols.txt"

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
    for q in user_input:
        ...