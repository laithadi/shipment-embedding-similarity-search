import pathlib

from ._generate_versioned_filename_for_outputs import generate_versioned_filename

DATA_PATH = pathlib.Path.cwd() / "data" / "shipment_dataset.csv"
USER_QUERY_INPUT = pathlib.Path.cwd() / "user_input" / "user_queries.txt"
USER_DF_COLS_INPUT = pathlib.Path.cwd() / "user_input" / "df_cols.txt"
OUTPUT_FILE_DIR = pathlib.Path.cwd() / "results" / "outputs" 
SIMILARITY_CALC_RES_DIR = pathlib.Path.cwd() / "results" / "similarity_calcs_res" 

o_filename = generate_versioned_filename(
    directory= OUTPUT_FILE_DIR,
    prefix= "results_",
    )

cs_filename = generate_versioned_filename(
    directory= SIMILARITY_CALC_RES_DIR,
    prefix= "detailed_summary_",
    )

OUTPUT_FILE_PATH = OUTPUT_FILE_DIR / o_filename
SIMILARITY_CALC_RES_PATH = SIMILARITY_CALC_RES_DIR / cs_filename

ORIGINAL_FILENAME_KEY = "original_filename"
VALUE_KEY = "value"
SCORE_KEY = "similarity_score"


__all__ = [
    "DATA_PATH",
    "USER_QUERY_INPUT",
    "USER_DF_COLS_INPUT",
    "OUTPUT_FILE_PATH",
    "SIMILARITY_CALC_RES_PATH",
]