import pathlib

from ._generate_versioned_filename_for_outputs import generate_versioned_filename

DATA_PATH = pathlib.Path.cwd() / "data" / "take_home_dataset.csv"
USER_QUERY_INPUT = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "user_queries.txt"
USER_DF_COLS_INPUT = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "df_cols.txt"
OUTPUT_FILE_DIR = pathlib.Path.cwd() / "src" / "playbooks" / "outputs" 
SIMILARITY_CALC_RES_DIR = pathlib.Path.cwd() / "src" / "playbooks" / "similarity_calcs_res" 

filename = generate_versioned_filename(directory= OUTPUT_FILE_DIR)

OUTPUT_FILE_PATH = OUTPUT_FILE_DIR / f"results_{filename}"
SIMILARITY_CALC_RES_PATH = SIMILARITY_CALC_RES_DIR / f"detailed_summary_{filename}"

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